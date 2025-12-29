from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage, QBrush
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import *
import os
from pathlib import Path
import subprocess
import sys
import argparse
import shutil
from multiprocessing import Process
import json
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from astropy.visualization.stretch import LogStretch, LinearStretch
from astropy.visualization import ImageNormalize

LOCAL_DIR = "GUI"
MAINDIR = Path(os.path.dirname(__file__).rpartition(LOCAL_DIR)[0])
sys.path.append(os.path.join(MAINDIR))
import imfit_run
import fit_monitor

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(250/100, 250/100), dpi=100)
        self.ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, galaxy_path, band, idx, fit_type, limits, cmap, stretch=LogStretch()):
        self.ax.cla()
        self.ax.set_axis_off()
        try:
            im = fits.getdata(os.path.join(galaxy_path, f"{fit_type}_{band}_composed.fits"))[idx]
            norm = ImageNormalize(stretch=stretch, vmin=limits[0], vmax=limits[1])
            self.ax.imshow(im, origin="lower", norm=norm, cmap=cmap)
        except:
            self.ax.text(0,0.5,"Cannot find FITs composed image!")
        finally:
            self.draw()

class DirOnlyChildrenFileSystemModel(QFileSystemModel):
    def __init__(self, mark_colors=None, galmarks=None, parent=None):
        super().__init__(parent)
        self.mark_colors = mark_colors or {}
        self.galmarks = galmarks or {}

    def hasChildren(self, index):
        # For invalid indices, fall back to default behavior
        if not index.isValid():
            return super().hasChildren(index)
        # Files are never considered to have children
        if not self.isDir(index):
            return False
        path = self.filePath(index)
        try:
            # Only report that this index has children if it contains subdirectories
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_dir():
                        return True
            return False
        except Exception:
            return super().hasChildren(index)

    def data(self, index, role):
        # Remove folder icon for leaf directories (galaxy folders)
        if role == QtCore.Qt.DecorationRole:
            try:
                # If it's a directory but has no subdirectories (a leaf), don't show an icon
                if self.isDir(index) and not self.hasChildren(index):
                    return None
            except Exception:
                pass

        # Color leaf directories based on galmarks
        if role == QtCore.Qt.BackgroundRole:
            try:
                if self.isDir(index) and not self.hasChildren(index):
                    name = Path(self.filePath(index)).name
                    mark = self.galmarks.get(name)
                    if mark:
                        col = self.mark_colors.get(mark)
                        if col:
                            return QBrush(QColor(col))
            except Exception:
                pass

        return super().data(index, role)

class MainWindow(QDialog):
    def __init__(self, galpathlist=None, p=None):
        super().__init__()
        ui_file = QFile(os.path.join(MAINDIR, LOCAL_DIR, 'fit_gui.ui'))
        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        # Loading the config file for the GUI
        with open(os.path.join(MAINDIR, LOCAL_DIR, 'config.json')) as config:
            self.gui_config = json.load(config)
            config.close()

        # Initializing widget types to make my autocomplete work
        self.currentgalaxytext: QTextBrowser = self.ui.currentgalaxytext
        self.config: QTextBrowser = self.ui.config
        self.params: QTextBrowser = self.ui.params

        # Initializing some variables
        self.galpathlist = galpathlist
        self.curr_gal_index = 0
        self.solvertype = "LM"
        self.band = "g"

        # Setting up the buttons
        self.ui.LMbutton.clicked.connect(lambda: self.set_solver("LM"))
        self.ui.NMbutton.clicked.connect(lambda: self.set_solver("NM"))
        self.ui.DEbutton.clicked.connect(lambda: self.set_solver("DE"))
        self.ui.gbutton.clicked.connect(lambda: self.set_band("g"))
        self.ui.gbutton.setShortcut(QKeySequence("g"))
        self.ui.rbutton.clicked.connect(lambda: self.set_band("r"))
        self.ui.rbutton.setShortcut(QKeySequence("r"))
        self.ui.ibutton.clicked.connect(lambda: self.set_band("i"))
        self.ui.ibutton.setShortcut(QKeySequence("i"))
        self.ui.zbutton.clicked.connect(lambda: self.set_band("z"))
        self.ui.zbutton.setShortcut(QKeySequence("z"))
        self.ui.markfitted.clicked.connect(lambda: self.markgalaxy("fitted"))
        self.ui.markfitted.setShortcut(QKeySequence("f"))
        self.ui.markreturn.clicked.connect(lambda: self.markgalaxy("return"))
        self.ui.markreturn.setShortcut(QKeySequence("e"))
        self.ui.markunable.clicked.connect(lambda: self.markgalaxy("unable"))
        self.ui.markunable.setShortcut(QKeySequence("u"))
        self.ui.saveconfigbutton.clicked.connect(self.saveconfig)

        self.ui.opends9button.clicked.connect(self.open_ds9)
        self.ui.opends9button.setShortcut(QKeySequence("O"))
        self.ui.refitbutton.clicked.connect(self.refit)
        self.ui.refitbutton.setShortcut(QKeySequence("CTRL+R"))
        self.ui.cancelbutton.clicked.connect(self.cancel)
        self.ui.cancelbutton.setShortcut(QKeySequence("CTRL+C"))

        # Get the fit type
        self.ui.fit_type_combo.currentTextChanged.connect(self.change_fit_type)
        self.fit_type = self.ui.fit_type_combo.currentText()

        # Loading the list of galaxies
        self.galaxytree: QTreeView = self.ui.galaxytree
        self.galaxytree.setColumnHidden(1, True)

        # Track the currently selected lowest-level galaxy folder (Path) if any
        self.selected_galaxy_path = None


        # Process list of currently running fits
        self.ps = []
        self.fit_dialogs = []

        # Setting up the FITs plots for the iamge, model, and residual
        self.img = PlotCanvas(parent=self.ui.galimg)
        self.model = PlotCanvas(parent=self.ui.galmodel)
        self.resid = PlotCanvas(parent=self.ui.galresid)

        # Loading the JSON file for the galaxy marks (whether fitted, need to return to, or can't fit)
        try:
            with open(os.path.join(MAINDIR, LOCAL_DIR, 'galmarks.json')) as f:
                self.galmarks = json.load(f)
        except:
            self.galmarks = {}

        # Ensure mark colors mapping exists
        self.colors = self.gui_config.get("mark_colors", {})

        # Use a QFileSystemModel that only reports directories as having children,
        # hides folder icons for leaf directories, and applies background colors.
        model = DirOnlyChildrenFileSystemModel(mark_colors=self.colors, galmarks=self.galmarks)
        model.setRootPath(str(p))
        self.galaxytree.setModel(model)
        self.galaxytree.setRootIndex(model.index(str(p)))
        self.galaxytree.setColumnHidden(1, True)
        self.galaxytree.setColumnHidden(2, True)
        self.galaxytree.setColumnHidden(3, True)
        # Detect selections on the tree to identify when a leaf directory is selected
        self.galaxytree.selectionModel().selectionChanged.connect(self.on_galaxytree_selection_changed)

        # Setting the colors of the marked galaxies
        # galnames = [os.path.basename(k["galname"]) for g in galpathlist.keys() for k in galpathlist[g]]
        # for gal in galnames:
        #     if gal in self.galmarks.keys():
        #         markas = self.galmarks[gal]
        #         i = self.galaxylist.findItems(gal, QtCore.Qt.MatchFlag.MatchContains | QtCore.Qt.MatchFlag.MatchRecursive, 0)[0]
        #         i.setBackground(0, QBrush(QColor(self.colors[markas])))
        #         self.galaxylist.repaint()
        # self.galaxylist.repaint()

        self.ui.show()

    def change_fit_type(self):
        self.fit_type = self.ui.fit_type_combo.currentText()
        self.changegal()

    def open_ds9(self, *args):
        p = self.selected_galaxy_path
        files = [f"{os.path.join(p, f'{self.fit_type}_{self.band}_composed.fits')}"]
        arg = ["ds9", "-cmap", self.gui_config["ds9_cmap"], "-scale", self.gui_config["ds9_scale"], "-scale", "limits", f"{self.gui_config["ds9_limits"][0]}", f"{self.gui_config["ds9_limits"][1]}", "-cube", "3"]
        arg.extend(files)
        subprocess.Popen(arg)
    
    def changegal(self):
        # Update UI based on the currently selected leaf galaxy folder
        galaxypath = self.selected_galaxy_path
        galaxy = galaxypath.name
        self.currentgalaxytext.setText(f"Current Galaxy: {galaxy}")
        self.currentgalaxytext.repaint()

        try:
            with open(os.path.join(galaxypath, f"{self.fit_type}_{self.band}.dat"), "r") as f:
                config_file = f.readlines()
            self.config.setPlainText("".join(config_file))
            self.config.repaint()
        except:
            self.config.setPlainText("Config file not found!")
            self.config.repaint()

        try:
            with open(os.path.join(galaxypath, f"{self.fit_type}_{self.band}_fit_params.txt"), "r") as f:
                config_file = f.readlines()
            self.params.setPlainText("".join(config_file))
            self.params.repaint()
        except:
            self.params.setPlainText("Fit Params not found!")
            self.params.repaint()

        pixmap = QPixmap(os.path.join(galaxypath, "image.jpg"))
        self.ui.galaxyjpg.setPixmap(pixmap)
        self.img.plot(galaxypath, 
                        self.band,
                        idx=0,
                        fit_type=self.fit_type, 
                        limits=self.gui_config["plot_limits"], 
                        cmap=self.gui_config["plot_cmap"])
        self.model.plot(galaxypath, 
                        self.band, 
                        idx=1, 
                        fit_type=self.fit_type, 
                        limits=self.gui_config["plot_limits"],
                        cmap=self.gui_config["plot_cmap"])
        self.resid.plot(galaxypath, 
                        self.band,
                        idx=2, 
                        fit_type=self.fit_type, 
                        limits=self.gui_config["plot_resid_limits"], 
                        cmap=self.gui_config["plot_resid_cmap"],
                        stretch=LinearStretch())
        
    def on_galaxytree_selection_changed(self, selected, deselected):
        """Called when the tree selection changes. If the selected item is a lowest-level
        directory (a directory with no subdirectories) we store it as the current galaxy
        and update the UI."""
        idxs = selected.indexes()
        if not idxs:
            return
        idx = idxs[0]
        model = self.galaxytree.model()
        # Only consider directory selections
        try:
            if not model.isDir(idx):
                return
        except Exception:
            return

        is_leaf = not model.hasChildren(idx)
        path = Path(model.filePath(idx))
        if is_leaf:
            self.selected_galaxy_path = path
            self.changegal()
        else:
            # Clear selection for non-leaf directories
            self.selected_galaxy_path = None

    def set_solver(self, solver):
        self.solvertype = solver 

    def set_band(self, band):
        self.band = band
        self.changegal(index=self.curr_gal_index)
    

    def refit(self):
        # Open a Fit Monitor dialog which runs IMFIT and streams stdout
        if getattr(self, "selected_galaxy_path", None) is None:
            print("No galaxy selected to refit")
            return
        path = self.selected_galaxy_path
        dlg = fit_monitor.FitMonitorDialog(path, self.band, self.solvertype, max_threads=self.gui_config["imfit_maxthreads"], fit_type=self.fit_type, parent=self)
        dlg.show()
        self.fit_dialogs.append(dlg)

        # Just refreshing the configs and stats and whatnot
        self.changegal()
    
    def cancel(self):
        # Try to cancel dialog-based fits first
        if len(self.fit_dialogs) > 0:
            dlg = self.fit_dialogs[-1]
            try:
                dlg.cancel()
                dlg.close()
            except Exception:
                pass
            self.fit_dialogs.pop()
            return

        if len(self.ps) > 0:
            self.ps[-1].terminate()
            self.ps.pop()
        else:
            print("No running IMFIT processes")
    
    def markgalaxy(self, markas):
        if getattr(self, "selected_galaxy_path", None) is None:
            return
        galname = self.selected_galaxy_path.name
        # Save mark state
        self.galmarks[galname] = markas
        with open(os.path.join(MAINDIR, LOCAL_DIR, 'galmarks.json'), 'w') as fp:
            json.dump(self.galmarks, fp)
        # Refresh the view so marked colors are updated
        try:
            model = self.galaxytree.model()
            model.layoutChanged.emit()
        except Exception:
            pass
    
    def saveconfig(self):
        if getattr(self, "selected_galaxy_path", None) is None:
            print("No galaxy selected to save config")
            return
        p = self.selected_galaxy_path
        new_config = self.ui.config.toPlainText()
        print(new_config)
        if not(new_config == ""):
            fit_type = self.ui.fit_type_combo.currentText()
            if os.path.isfile(os.path.join(p, f"{fit_type}_{self.band}.dat")):
                shutil.copyfile(src=os.path.join(p, f"{fit_type}_{self.band}.dat"), dst=os.path.join(p, f"{fit_type}_{self.band}.dat.bak"))
            with open(os.path.join(p, f"{fit_type}_{self.band}.dat"), "w") as f:
                f.write(new_config)

def get_galaxies(p):
    structure = os.walk(p)

    gal_pathdict = {}
    for root, dirs, files in structure:
        if not(files == []):
            # Assumes data is at the end of the file tree
            galpath = Path(root)
            if galpath != None:
                gal_pathdict[galpath.name] = galpath

    return gal_pathdict, p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Open IMFIT helper GUI"
    )
    
    parser.add_argument("-p", help="Path to folder containing galaxies", default=".")

    args = parser.parse_args()
    p = Path(args.p).resolve()
    app = QApplication(sys.argv)
    r = get_galaxies(p)
    main_win = MainWindow(p, r[1])
    sys.exit(app.exec())
