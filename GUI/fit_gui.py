from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QFile, QIODevice
from PySide6.QtWidgets import *
from PyQt6 import uic
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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from astropy.visualization.stretch import LogStretch
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

    def plot(self, galaxy_path, band, idx, fit_type):
        self.ax.cla()
        self.ax.set_axis_off()
        try:
            im = fits.getdata(os.path.join(galaxy_path, f"{fit_type}_{band}_composed.fits"))[idx]
            norm = ImageNormalize(stretch=LogStretch(), vmin=0, vmax=1)
            self.ax.imshow(im, origin="lower", norm=norm, cmap="inferno")
        except:
            self.ax.text(0,0.5,"Cannot find FITs composed image!")
        finally:
            self.draw()

class MainWindow(QDialog):
    def __init__(self, galpathlist=None):
        super().__init__()
        ui_file = QFile(os.path.join(MAINDIR, LOCAL_DIR, 'fit_gui.ui'))
        loader = QUiLoader()
        self.ui = loader.load(ui_file)

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
        self.ui.nextgalbutton.clicked.connect(self.next_galaxy)
        self.ui.refitbutton.clicked.connect(self.refit)
        self.ui.refitbutton.setShortcut(QKeySequence("CTRL+R"))
        self.ui.cancelbutton.clicked.connect(self.cancel)
        self.ui.cancelbutton.setShortcut(QKeySequence("CTRL+C"))

        # Get the fit type
        self.ui.fit_type_combo.currentTextChanged.connect(self.change_fit_type)
        self.fit_type = self.ui.fit_type_combo.currentText()

        # Loading the list of galaxies
        self.galaxylist: QTreeWidget = self.ui.galaxylist
        # self.galaxylist.addItems([os.path.basename(g) for g in galpathlist])
        self.galaxylist.itemSelectionChanged.connect(self.changegal)
        # print(self.galpathlist)
        for galtype in self.galpathlist.keys():
            a = QtWidgets.QTreeWidgetItem(self.galaxylist, [galtype])
            for galpath_dict in self.galpathlist[galtype]:
                print(galpath_dict)
                b = QtWidgets.QTreeWidgetItem([str(galpath_dict["galname"])])
                a.addChild(b)

        self.colors = {
            "fitted" : "#35bd49",
            "return" : "#bdba35",
            "unable" : "#bd3535"
        }

        # Process list of currently running fits
        self.ps = []
        self.fit_dialogs = []

        # Setting up the FITs plots for the iamge, model, and residual
        self.img = PlotCanvas(parent=self.ui.galimg)
        self.model = PlotCanvas(parent=self.ui.galmodel)
        self.resid = PlotCanvas(parent=self.ui.galresid)

        # Switch over to the first galaxy
        # self.changegal(self.curr_gal_index)

        # Loading the JSON file for the galaxy marks (whether fitted, need to return to, or can't fit)
        try:
            with open('galmarks.json') as f:
                self.galmarks = json.load(f)
        except:
            self.galmarks = {}

        # Setting the colors of the marked galaxies
        # galnames = [os.path.basename(g) for g in galpathlist]
        # for gal in galnames:
        #     if gal in self.galmarks.keys():
        #         markas = self.galmarks[gal]
        #         self.galaxylist.item(galnames.index(gal)).setBackground(QColor(self.colors[markas]))
        # self.galaxylist.repaint()

        self.ui.show()

    def change_fit_type(self):
        self.fit_type = self.ui.fit_type_combo.currentText()
        self.changegal(index=self.curr_gal_index)

    def open_ds9(self, Dialog):
        print("Opening DS9...")
        p = self.galpathlist[self.curr_gal_index]

        # files = [os.path.join(p, f"image_{b}.fits") for b in "griz"]
        files = [f"{os.path.join(p, f'{self.fit_type}_{self.band}_composed.fits')}"]
        arg = ["ds9", "-cmap", "inferno", "-scale", "log", "-scale", "limits", "0", "10", "-cube", "3"]
        arg.extend(files)
        subprocess.Popen(arg)
    
    def next_galaxy(self):
        self.curr_gal_index += 1
        self.changegal(self.curr_gal_index)
    
    def changegal(self):
        print(self.galaxylist.currentItem().text(0))
        print(self.galaxylist.topLevelItem(0))
        # print(self.galaxylist.currentIndex().internalId())
        print("Hello")
    # def changegal(self, index=None):
    #     if index == None:
    #         galaxy = self.galaxylist.selectedItems()[0].text()
    #         gl = [os.path.basename(g) for g in self.galpathlist]
    #         self.curr_gal_index = gl.index(galaxy)
    #     self.currentgalaxytext.setText(f"Current Galaxy: {os.path.basename(self.galpathlist[self.curr_gal_index])}")
    #     self.currentgalaxytext.repaint()

    #     p = self.galpathlist[self.curr_gal_index]
    #     try:
    #         with open(os.path.join(p, f"{self.fit_type}_{self.band}.dat"), "r") as f:
    #             config_file = f.readlines()
    #         self.config.setPlainText("".join(config_file))
    #         self.config.repaint()
    #     except:
    #         self.config.setPlainText("Config file not found!")
    #         self.config.repaint()

    #     try:
    #         with open(os.path.join(p, f"{self.fit_type}_{self.band}_fit_params.txt"), "r") as f:
    #             config_file = f.readlines()
    #         self.params.setPlainText("".join(config_file))
    #         self.params.repaint()
    #     except:
    #         self.params.setPlainText("Fit Params not found!")
    #         self.params.repaint()

    #     pixmap = QPixmap(os.path.join(p, "image.jpg"))
    #     self.ui.galaxyjpg.setPixmap(pixmap)
    #     self.img.plot(self.galpathlist[self.curr_gal_index], self.band, idx=0, fit_type=self.fit_type)
    #     self.model.plot(self.galpathlist[self.curr_gal_index], self.band, idx=1, fit_type=self.fit_type)
    #     self.resid.plot(self.galpathlist[self.curr_gal_index], self.band, idx=2, fit_type=self.fit_type)
         
    def set_solver(self, solver):
        self.solvertype = solver 

    def set_band(self, band):
        self.band = band
        self.changegal(index=self.curr_gal_index)
    

    def refit(self):
        # Open a Fit Monitor dialog which runs IMFIT and streams stdout
        path = self.galpathlist[self.curr_gal_index]
        dlg = fit_monitor.FitMonitorDialog(path, self.band, self.solvertype, max_threads=8, fit_type=self.fit_type, parent=self)
        dlg.show()
        self.fit_dialogs.append(dlg)

        # Just refreshing the configs and stats and whatnot
        self.changegal(index=self.curr_gal_index)
    
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
        self.galaxylist.item(self.curr_gal_index).setBackground(QColor(self.colors[markas]))
        self.galaxylist.repaint()
        
        self.galmarks[self.galaxylist.item(self.curr_gal_index).text()] = markas
        with open('galmarks.json', 'w') as fp:
            json.dump(self.galmarks, fp)
    
    def saveconfig(self):
        p = self.galpathlist[self.curr_gal_index]
        new_config = self.ui.config.toPlainText()
        print(new_config)
        if not(new_config == ""):
            fit_type = self.fit_type_combo.currentText()
            if os.path.isfile(os.path.join(p, f"{fit_type}_{self.band}.dat")):
                shutil.copyfile(src=os.path.join(p, f"{fit_type}_{self.band}.dat"), dst=os.path.join(p, f"{fit_type}_{self.band}.dat.bak"))
            with open(os.path.join(p, f"{fit_type}_{self.band}.dat"), "w") as f:
                f.write(new_config)

def get_galaxies(p):
    structure = os.walk(p)

    gal_pathlist = []
    gal_pathdict = {}
    for root, dirs, files in structure:
        if not(files == []):
            # Assumes data is at the end of the file tree
            galpath = Path(root)
            if galpath != None:
                # gal_pathlist.append([galpath.parent.name, galpath])
                try:
                    # gal_pathdict[galpath.parent.name].append(galpath)
                    gal_pathdict[galpath.parent.name].append({"galname": galpath.name, "galpath": galpath})
                except:
                    gal_pathdict[galpath.parent.name] = [{"galname": galpath.name, "galpath": galpath}]

    return gal_pathdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Open IMFIT helper GUI"
    )
    
    parser.add_argument("-p", help="Path to folder containing galaxies", default=".")

    args = parser.parse_args()
    p = Path(args.p).resolve()
    app = QApplication(sys.argv)
    main_win = MainWindow(get_galaxies(p))
    sys.exit(app.exec())
