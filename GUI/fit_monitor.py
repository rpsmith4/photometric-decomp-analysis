from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage, QTextCursor
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import *
import os
import sys
from pathlib import Path
LOCAL_DIR = "GUI"
MAINDIR = Path(os.path.dirname(__file__).rpartition(LOCAL_DIR)[0])
sys.path.append(os.path.join(MAINDIR))
import imfit_run

IMAN_DIR = Path(os.path.dirname(__file__))
sys.path.append(os.path.join(IMAN_DIR, 'iman_new/decomposition/make_model'))
import make_model_ima_imfit

class FitWorker(QtCore.QThread):
    output = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    def __init__(self, path, band, solver, max_threads, fit_type, mask=True, psf=True, invvar=True, parent=None):
        super().__init__(parent)
        self.path = str(Path(path).resolve())
        self.band = band
        self.solver = solver
        self.max_threads = max_threads
        self.fit_type = fit_type
        self.mask = mask
        self.psf = psf
        self.invvar = invvar

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
            imfit_run.run_imfit(self.band, mask=self.mask, psf=self.psf, invvar=self.invvar,
                                alg=self.solver, max_threads=self.max_threads, fit_type=self.fit_type,
                                stdout_callback=cb)
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
            params_file = f"{self.fit_type}_{self.band}_fit_params.txt"
            img_file = f"image_{self.band}.fits"
            psf_file = f"psf_patched_{self.band}.fits"
            mask_file = "image_mask.fits"

            if os.path.exists(params_file):

                if self.mask and os.path.exists(mask_file):
                    from astropy.io import fits
                    img_dat = fits.open(img_file)
                    img = img_dat[0].data
                    mask_img = fits.open(mask_file)[0].data
                    img = img * (1 - mask_img)
                    fits.writeto("masked.fits", data=img, header=img_dat[0].header, overwrite=True)
                    make_model_ima_imfit.main("masked.fits", params_file, psf_file, composed_model_file=f"{self.fit_type}_{self.band}_composed.fits", comp_names=["Host", "Polar"])
                    try:
                        os.remove("./masked.fits")
                    except Exception:
                        pass
                else:
                    make_model_ima_imfit.main(img_file, params_file, psf_file, composed_model_file=f"{self.fit_type}_{self.band}_composed.fits", comp_names=["Host", "Polar"])

        except Exception as e:
            self.output.emit(f"Warning: failed to make composed image: {e}\n")

        try:
            os.chdir(cwd)
        except:
            pass

        self.finished.emit(0)


class FitMonitorDialog:
    def __init__(self, path, band, solver, max_threads=8, fit_type="2_sersic", parent=None):
        self.parent = parent
        ui_file = QFile(os.path.join(MAINDIR, LOCAL_DIR, 'fit_monitor.ui'))
        loader = QUiLoader()
        self.ui = loader.load(ui_file)

        self.path = path
        self.band = band
        self.solver = solver
        self.max_threads = max_threads
        self.fit_type = fit_type

        # UI elements from the .ui
        self.ui.stdoutEdit.setReadOnly(True)
        self.ui.cancelButton.clicked.connect(self.cancel)
        self.ui.closeButton.clicked.connect(self.close)

        # Worker thread
        self.worker = FitWorker(path, band, solver, max_threads, fit_type)
        self.worker.output.connect(self._append_output)
        self.worker.finished.connect(self._finished)

        self.ui.titleLabel.setText(f"IMFIT: {os.path.basename(self.path)}  band={self.band}  solver={self.solver}")
        self.ui.statusLabel.setText("Status: Running")

        # Start
        self.worker.start()

    def _append_output(self, text):
        # Append text to stdout view
        self.ui.stdoutEdit.moveCursor(QTextCursor.MoveOperation.End)
        self.ui.stdoutEdit.insertPlainText(text)
        self.ui.stdoutEdit.moveCursor(QTextCursor.MoveOperation.End)

    def _finished(self, code):
        self.ui.statusLabel.setText(f"Status: Finished (code={code})")
        # If parent is the main window, trigger a refresh of the currently selected galaxy
        try:
            parent = self.parent
            if parent is not None and hasattr(parent, 'changegal'):
                try:
                    parent.changegal(index=parent.curr_gal_index)
                except Exception:
                    # best-effort: if the parent's index attribute is named differently, try calling without args
                    try:
                        parent.changegal()
                    except Exception:
                        pass
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

