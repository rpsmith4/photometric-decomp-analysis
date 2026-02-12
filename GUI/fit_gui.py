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
import pyimfit
import shutil
import numpy as np

LOCAL_DIR = "GUI"
MAINDIR = Path(os.path.dirname(__file__).rpartition(LOCAL_DIR)[0])
sys.path.append(os.path.join(MAINDIR, "decomposer"))
import imfit_run
import fit_monitor

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(250/100, 250/100), dpi=100)
        self.ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, im, limits, cmap, stretch=LogStretch()):
        self.ax.cla()
        self.ax.set_axis_off()
        if im.any():
            norm = ImageNormalize(stretch=stretch, vmin=limits[0], vmax=limits[1])
            self.ax.imshow(im, origin="lower", norm=norm, cmap=cmap)
        else:
            self.ax.text(0,0.5,"Cannot find FITs image!")
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

class ParamSliderWidget(QWidget):
    def __init__(self, paramname, initval, lowlim, hilim, fixed=False, ndigits=3, parent=None):
        super().__init__(parent)
        self.paramname = paramname
        self.ndigits = ndigits
        self.scale = 10 ** ndigits
        self.fixed = fixed

        parameter_adjust_layout = QHBoxLayout()

        self.text = QTextBrowser()
        self.text.setText(str(paramname))
        self.text.setFixedSize(50, 30)
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.fixed_checkbox = QCheckBox("Fixed")
        self.fixed_checkbox.setChecked(fixed)

        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(int(lowlim * self.scale), int(hilim * self.scale))
        self.slider.setTickInterval(5)
        self.slider.setValue(int(initval * self.scale))

        parameter_adjust_layout.addWidget(self.text)
        parameter_adjust_layout.addWidget(self.slider)
        parameter_adjust_layout.addWidget(self.fixed_checkbox)

        spinboxes_layout = QHBoxLayout()
        self.minspinbox = QDoubleSpinBox()
        self.minspinbox.setDecimals(ndigits)
        self.minspinbox.setMaximum(hilim)
        self.minspinbox.setMinimum(-1e9)
        self.minspinbox.setValue(lowlim)
        self.minspinbox.setMaximumWidth(65)

        self.valspinbox = QDoubleSpinBox()
        self.valspinbox.setDecimals(ndigits)
        self.valspinbox.setMaximum(hilim)
        self.valspinbox.setMinimum(lowlim)
        self.valspinbox.setValue(initval)
        self.valspinbox.setMaximumWidth(65)

        self.maxspinbox = QDoubleSpinBox()
        self.maxspinbox.setDecimals(ndigits)
        self.maxspinbox.setMinimum(lowlim)
        self.maxspinbox.setMaximum(1e9)
        self.maxspinbox.setValue(hilim)
        self.maxspinbox.setMaximumWidth(65)

        spinboxes_layout.addWidget(self.minspinbox)
        spinboxes_layout.addWidget(self.valspinbox)
        spinboxes_layout.addWidget(self.maxspinbox)

        parameter_adjust_layout.addLayout(spinboxes_layout)
        self.setLayout(parameter_adjust_layout)

        self.set_fixed_state(fixed)

        self.slider.valueChanged.connect(self.slider_changed)
        self.valspinbox.valueChanged.connect(self.spinbox_changed)
        self.minspinbox.valueChanged.connect(self.minspinbox_changed)
        self.maxspinbox.valueChanged.connect(self.maxspinbox_changed)
        self.fixed_checkbox.stateChanged.connect(lambda state: self.set_fixed_state(state==2))

    def set_fixed_state(self, is_fixed):
        self.minspinbox.setEnabled(not is_fixed)
        self.maxspinbox.setEnabled(not is_fixed)
        # self.slider.setEnabled(not is_fixed)
        # self.valspinbox.setEnabled(not is_fixed)

    def slider_changed(self, value):
        float_val = value / self.scale
        self.valspinbox.blockSignals(True)
        self.valspinbox.setValue(float_val)
        self.valspinbox.blockSignals(False)

    def spinbox_changed(self, value):
        int_val = int(round(value * self.scale))
        self.slider.blockSignals(True)
        self.slider.setValue(int_val)
        self.slider.blockSignals(False)

    def minspinbox_changed(self, new_min):
        self.slider.setMinimum(int(new_min * self.scale))
        self.valspinbox.setMinimum(new_min)
        self.maxspinbox.setMinimum(new_min)
        if self.valspinbox.value() < new_min:
            self.valspinbox.setValue(new_min)
        if self.slider.value() < int(new_min * self.scale):
            self.slider.setValue(int(new_min * self.scale))

    def maxspinbox_changed(self, new_max):
        self.slider.setMaximum(int(new_max * self.scale))
        self.valspinbox.setMaximum(new_max)
        self.minspinbox.setMaximum(new_max)
        if self.valspinbox.value() > new_max:
            self.valspinbox.setValue(new_max)
        if self.slider.value() > int(new_max * self.scale):
            self.slider.setValue(int(new_max * self.scale))

    def get_values(self):
        return {
            'value': self.valspinbox.value(),
            'min': self.minspinbox.value(),
            'max': self.maxspinbox.value(),
            'fixed': self.fixed_checkbox.isChecked()
        }

class MainWindow(QMainWindow):
    def __init__(self, p=None):
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
        # self.config: QTextBrowser = self.ui.config
        self.params: QTextBrowser = self.ui.params

        # Initializing some variables
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
        self.configimg = PlotCanvas(parent=self.ui.configimg)
        self.configresid = PlotCanvas(parent=self.ui.configresid)

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
    
    def get_composed_data(self, galaxy_path, band, idx, fit_type):
        # idx = 0 -> Image with mask, 1 -> Model image, 2 -> Residual, 3 -> Percent residual, 4 Onwards -> Components of model
        try:
            im = fits.getdata(os.path.join(galaxy_path, f"{fit_type}_{band}_composed.fits"))[idx]
        except:
            im = np.array([])
        return im

    def getconfigim(self, galaxypath, band, fit_type, shape, maxThreads=4):
        try:
            model_desc = pyimfit.parse_config_file(os.path.join(galaxypath, f"{fit_type}_{band}.dat"))
            psf = fits.getdata(os.path.join(galaxypath, f"psf_patched_{band}.fits"))
            # imfitter = pyimfit.Imfit(model_desc, psf=psf)
            imfitter = pyimfit.Imfit(model_desc, maxThreads=maxThreads)
            im = imfitter.getModelImage(shape=shape)
        except:
            im = np.array([])

        return im
    
    def getconfigresid(self, im, imconfig):
        return im - imconfig


    def changegal(self):
        # Store the current config model for later editing
        self.current_config_model = None
        # Update UI based on the currently selected leaf galaxy folder
        galaxypath = self.selected_galaxy_path
        galaxy = galaxypath.name
        self.currentgalaxytext.setText(f"Current Galaxy: {galaxy}")
        self.currentgalaxytext.repaint()

        config_path = os.path.join(galaxypath, f"{self.fit_type}_{self.band}.dat")
        config_model = pyimfit.parse_config_file(config_path)
        self.current_config_model = config_model
        config_dict = config_model.getModelAsDict()
        function_list = config_dict["function_sets"][0]["function_list"]
        layout: QVBoxLayout = self.ui.configsliders
        # Reset the layout first
        try:
            self.clearLayout(layout)
        except Exception as e:
            print(e)
            pass
        for func_idx, func in enumerate(function_list):
            params = func["parameters"]
            for param in params.keys():
                initval = params[param][0]
                fixed = False
                if params[param][1] == 'fixed':
                    lowlim = initval
                    hilim = initval
                    fixed = True
                else:
                    lowlim = params[param][1]
                    hilim = params[param][2]

                # Use (func_idx, param) as key to distinguish duplicate param names
                self.draw_params(initval, lowlim, hilim, fixed, (func_idx, param), layout)


        try:
            with open(os.path.join(galaxypath, f"{self.fit_type}_{self.band}_fit_params.txt"), "r") as f:
                params_file = f.readlines()
            self.params.setPlainText("".join(params_file))
            self.params.repaint()
        except:
            self.params.setPlainText("Fit Params not found!")
            self.params.repaint()

        pixmap = QPixmap(os.path.join(galaxypath, "image.jpg"))
        self.ui.galaxyjpg.setPixmap(pixmap)
        

        # self.img.get_composed_data(galaxypath, self.band, idx=0, fit_type=self.fit_type)
        img = self.get_composed_data(galaxypath, self.band, idx=0, fit_type=self.fit_type)
        self.img.plot(img, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"])

        model = self.get_composed_data(galaxypath, self.band, idx=1, fit_type=self.fit_type)
        self.model.plot(model, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"])

        resid = self.get_composed_data(galaxypath, self.band, idx=2, fit_type=self.fit_type)
        self.resid.plot(resid, limits=self.gui_config["plot_resid_limits"], cmap=self.gui_config["plot_resid_cmap"], stretch=LinearStretch())
        
        imconfig = self.getconfigim(galaxypath, self.band, self.fit_type, np.shape(img), maxThreads=self.gui_config["imfit_maxthreads"])
        self.configimg.plot(imconfig, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"])

        imresidconfig = self.getconfigresid(img, imconfig)
        self.configresid.plot(imresidconfig, limits=self.gui_config["plot_resid_limits"], cmap=self.gui_config["plot_resid_cmap"], stretch=LinearStretch())

    def clearLayout(self, layout):
        if isinstance(layout, QLayout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                    # layout.removeWidget(widget)
                else:
                    self.clearLayout(item.layout())
                    # layout.removeItem(item)
 
    def draw_params(self, initval, lowlim, hilim, fixed, paramkey, layout):
        if not hasattr(self, 'param_widgets'):
            self.param_widgets = {}
        func_idx, paramname = paramkey
        ndigits = 3
        widget = ParamSliderWidget(paramname, initval, lowlim, hilim, fixed=fixed, ndigits=ndigits)
        layout.addWidget(widget)
        self.param_widgets[paramkey] = widget
        
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
        self.changegal()

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
        fit_type = self.ui.fit_type_combo.currentText()
        config_path = os.path.join(p, f"{fit_type}_{self.band}.dat")

        # If we have a config model and param_widgets, update the config model with the new values
        if hasattr(self, 'current_config_model') and hasattr(self, 'param_widgets') and self.current_config_model is not None:
            model_dict = self.current_config_model.getModelAsDict()
            function_list = model_dict["function_sets"][0]["function_list"]
            # Update parameter values from widgets
            for func_idx, func in enumerate(function_list):
                params = func["parameters"]
                for param in params.keys():
                    key = (func_idx, param)
                    if key in self.param_widgets:
                        values = self.param_widgets[key].get_values()
                        if values['fixed']:
                            # Only value and 'fixed' string
                            params[param] = [values['value'], 'fixed']
                        else:
                            # Value, min, max
                            params[param] = [values['value'], values['min'], values['max']]
            # Rebuild the config description from the updated dict
            new_model = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict)
            config_text = "".join(new_model.getStringDescription())
            # Backup old config
            if os.path.isfile(config_path):
                shutil.copyfile(src=config_path, dst=config_path + ".bak")
            with open(config_path, "w") as f:
                f.write(config_text)
        else:
            # Fallback: use the text in the config editor if present
            new_config = self.ui.config.toPlainText()
            print(new_config)
            if not(new_config == ""):
                if os.path.isfile(config_path):
                    shutil.copyfile(src=config_path, dst=config_path + ".bak")
                with open(config_path, "w") as f:
                    f.write(new_config)
        if f: f.close()

        # Refresh config image and residual
        self.changegal()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Open IMFIT helper GUI"
    )
    
    parser.add_argument("-p", help="Path to folder containing galaxies", default=".")

    args = parser.parse_args()
    p = Path(args.p).resolve()
    app = QApplication(sys.argv)
    main_win = MainWindow(p)
    sys.exit(app.exec())
