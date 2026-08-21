from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage, QTextCursor
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import *
import os
import sys
from pathlib import Path
import html
import math
import re
LOCAL_DIR = "GUI"
MAINDIR = Path(os.path.dirname(__file__).rpartition(LOCAL_DIR)[0])
sys.path.append(os.path.join(MAINDIR))
import imfit_run
import pyimfit
from utils import DataSet
from threading import Thread
import traceback as tb

IMAN_DIR = Path(os.path.dirname(__file__))
sys.path.append(os.path.join(IMAN_DIR, 'iman_new/decomposition/make_model'))
import make_model_ima_imfit

class FitWorker(QtCore.QThread):
    output = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    # def __init__(self, path, band, solver, max_threads, fit_type, mask=True, psf=True, invvar=True, config_file=None, gui_config=None, fit_params_path=None, composed_image_path=None, parent=None):
    def __init__(self, path, band, solver, max_threads, dataset: DataSet, fit_type, use_mask=True, use_psf=True, use_invvar=True, gui_config=None, parent=None):
        super().__init__(parent)
        self.path = str(Path(path).resolve())
        self.band = band
        self.solver = solver
        self.max_threads = max_threads
        self.fit_type = fit_type
        self.use_mask = use_mask
        self.use_psf = use_psf
        self.use_invvar = use_invvar
        self.gui_config = gui_config
        self.dataset = dataset

    def run(self):
        # Change to target directory and run imfit, streaming stdout
        try:
            cwd = os.getcwd()
            os.chdir(self.path)
        except Exception as e:
            self.output.emit(f"Error changing directory: {e}\n")
            self.finished.emit(-1)
            return

        def cb(line):
            self.output.emit(line)

        # Call the low-level runner with the callback so we can stream stdout
        try:
            # imfit_run.run_imfit(self.band, mask=self.mask, psf=self.psf, invvar=self.invvar,
            #                     alg=self.solver, max_threads=self.max_threads, fit_type=self.fit_type,
            #                     config_file=self.config_file, gui_config=self.gui_config,
            #                     stdout_callback=cb, params_file=self.fit_params_path)

            imfit_run.run_imfit(self.band, mask=self.use_mask, psf=self.use_psf, invvar=self.use_invvar,
                                alg=self.solver, max_threads=self.max_threads, fit_type=self.fit_type,
                                config_file=self.dataset.config_path, 
                                gui_config=self.gui_config, image_file=self.dataset.fits_image_path, invvar_file=self.dataset.fits_invvar_image_path, psf_file=self.dataset.fits_psf_path, mask_file=self.dataset.mask_path,
                                stdout_callback=cb, params_file=self.dataset.fit_results_path)
            
            # init_model = pyimfit.ModelDescription.load(self.dataset.config_path)
            # imfit_fitter = pyimfit.Imfit(init_model, self.dataset.fits_psf, maxThreads= self.max_threads)
            # fit_result = target=imfit_fitter.fit(self.dataset.fits_image, self.dataset.fits_invvar_image, self.dataset.fits_mask, solver=self.solver,error_type="weight",verbose=1)
            # model_desc_str = imfit_fitter.getModelDescription().getStringDescription()
            # print(model_desc_str)
            # print(init_model.getStringDescription())
            # with open(self.dataset.config_path, 'w') as f:
            #     for line in model_desc_str:
            #         f.write(f"{line}\n")

        except Exception as e:
            self.output.emit(f"Error running imfit: {e}\n")
            try:
                os.chdir(cwd)
            except:
                pass
            self.finished.emit(-1)
            return

        # After imfit finishes, attempt to make composed image (similar to imfit_run.main behavior)
        try:
            params_file = os.path.basename(self.dataset.fit_results_path) 
            composed_file = os.path.basename(self.dataset.fits_composed_path)
            img_file = os.path.basename(self.dataset.fits_image_path) 
            psf_file = os.path.basename(self.dataset.fits_psf_path) 
            mask_file = os.path.basename(self.dataset.mask_path) 
            if self.gui_config["imfit_path"] != "":
                imfitPath = str(Path(self.gui_config["imfit_path"])) + "/"
            else:
                imfitPath = ""

            if os.path.exists(params_file):

                if self.use_mask and os.path.exists(mask_file):
                    from astropy.io import fits
                    img_dat = fits.open(img_file)
                    img = img_dat[0].data
                    mask_img = fits.open(mask_file)[0].data
                    img = img * (1 - mask_img)
                    fits.writeto("masked.fits", data=img, header=img_dat[0].header, overwrite=True)
                    
                    make_model_ima_imfit.main("masked.fits", params_file, psf_file, composed_model_file=composed_file, comp_names=self.dataset.func_labels, imfitPath=imfitPath, mask=mask_img)
                    try:
                        os.remove("./masked.fits")
                    except Exception:
                        pass
                else:
                    make_model_ima_imfit.main(img_file, params_file, psf_file, composed_model_file=composed_file, comp_names=self.dataset.func_labels, imfitPath=imfitPath)

        except Exception as e:
            self.output.emit(f"Warning: failed to make composed image: {e}\n")
            print(tb.format_exc())

        try:
            os.chdir(cwd)
        except:
            pass

        self.finished.emit(0)


class FitMonitorDialog:
    # def __init__(self, path, band, solver, max_threads=8, fit_type="2_sersic", config_file=None, gui_config=None, fit_params_path=None, composed_image_path=None, parent=None):
    def __init__(self, path, band, solver, dataset, max_threads=8, fit_type="2_sersic", gui_config=None, parent=None):
        self.parent = parent
        ui_file = QFile(os.path.join(MAINDIR, LOCAL_DIR, 'fit_monitor.ui'))
        loader = QUiLoader()
        self.ui = loader.load(ui_file)

        self.path = path
        self.band = band
        self.solver = solver
        self.max_threads = max_threads
        self.fit_type = fit_type

        self.dataset = dataset
        self.config_file = dataset.config_path

        self._stdout_buffer = ""
        self._current_func_idx = -1
        self._config_bounds = self._load_config_bounds(self.config_file)

        # UI elements from the .ui
        self.ui.stdoutEdit.setReadOnly(True)
        self.ui.cancelButton.clicked.connect(self.cancel)
        self.ui.closeButton.clicked.connect(self.close)

        # Worker thread
        self.worker = FitWorker(path, band, solver, max_threads, self.dataset, fit_type, gui_config=gui_config)
        self.worker.output.connect(self._append_output)
        self.worker.finished.connect(self._finished)

        self.ui.titleLabel.setText(f"IMFIT: {os.path.basename(self.path)}  band={self.band}  solver={self.solver}")
        self.ui.statusLabel.setText("Status: Running")

        # Start
        self.worker.start()

    def _append_output(self, text):
        # Append text to stdout view with highlighting
        self._stdout_buffer += text
        lines = self._stdout_buffer.split("\n")
        if not text.endswith("\n"):
            self._stdout_buffer = lines.pop()
        else:
            self._stdout_buffer = ""
            if lines and lines[-1] == "":
                lines.pop()

        for line in lines:
            self._maybe_update_function_index(line)
            formatted = self._format_monitor_line(line)
            self.ui.stdoutEdit.moveCursor(QTextCursor.MoveOperation.End)
            self.ui.stdoutEdit.insertHtml(formatted + "<br />")
            self.ui.stdoutEdit.moveCursor(QTextCursor.MoveOperation.End)

    def _maybe_update_function_index(self, line):
        if "FUNCTION" in line:
            self._current_func_idx += 1

    def _load_config_bounds(self, config_file):
        if config_file is None or not os.path.exists(config_file):
            return None
        try:
            config = pyimfit.parse_config_file(config_file)
            config_dict = config.getModelAsDict()
            function_list = config_dict["function_sets"][0]["function_list"]
            bounds = []
            for func in function_list:
                param_bounds = {}
                for pname, pval in func["parameters"].items():
                    if pval[1] == 'fixed':
                        param_bounds[pname] = (pval[0], pval[0])
                    else:
                        param_bounds[pname] = (pval[1], pval[2])
                bounds.append(param_bounds)
            return bounds
        except Exception:
            return None

    def _format_monitor_line(self, line):
        escaped = html.escape(line).replace(" ", "&nbsp;")
        highlight = False
        if "+/-" in line and self._config_bounds is not None and self._current_func_idx >= 0 and self._current_func_idx < len(self._config_bounds):
            match = re.match(r"^\s*(\S+)\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*#\s*\+/-\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", line)
            if match:
                pname = match.group(1)
                try:
                    pval = float(match.group(2))
                    punc = float(match.group(3))
                except ValueError:
                    pval = None
                    punc = None
                if pval is not None and punc == 0:
                    bounds = self._config_bounds[self._current_func_idx].get(pname)
                    if bounds is not None:
                        lowlim, hilim = bounds
                        if lowlim != hilim: # Parameter is fixed is these are equal
                            highlight = True
        if highlight:
            return f"<span style='background-color:yellow; color:red;'>{escaped}</span>"
        return escaped

    def _finished(self, code):
        self.ui.statusLabel.setText(f"Status: Finished (code={code})")
        # If parent is the main window, trigger a refresh of the currently selected galaxy
        try:
            parent = self.parent
            if parent is not None and hasattr(parent, 'refitdone'):
                parent.refitdone()
        except Exception:
            pass

    def cancel(self):
        # Try to terminate the running imfit process
        try:
            imfit_run.terminate_imfit()
            self.ui.statusLabel.setText("Status: Terminated")
        except Exception as e:
            self.ui.stdoutEdit.insertPlainText(f"Failed to terminate: {e}\n")

    def show(self):
        self.ui.show()

    def close(self):
        self.ui.close()

