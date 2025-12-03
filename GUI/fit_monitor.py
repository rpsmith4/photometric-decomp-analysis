from PyQt6 import QtCore, QtWidgets, uic
from PyQt6.QtWidgets import QDialog
from PyQt6.QtGui import QTextCursor
import os
import sys
from pathlib import Path
LOCAL_DIR = "GUI"
MAINDIR = Path(os.path.dirname(__file__).rpartition(LOCAL_DIR)[0])
sys.path.append(os.path.join(MAINDIR))
import imfit_run

class FitWorker(QtCore.QThread):
    output = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(int)

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
                # import make_model helper (imfit_run already appends IMAN path; replicate minimal import)
                IMAN_DIR = os.path.expanduser("~/Documents/iman_new")
                sys.path.append(os.path.join(IMAN_DIR, 'decomposition/make_model'))
                import make_model_ima_imfit

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


class FitMonitorDialog(QDialog):
    def __init__(self, path, band, solver, max_threads=8, fit_type="2_sersic", parent=None):
        super().__init__(parent)
        ui_path = os.path.join(os.path.dirname(__file__), 'fit_monitor.ui')
        uic.loadUi(ui_path, self)

        self.path = path
        self.band = band
        self.solver = solver
        self.max_threads = max_threads
        self.fit_type = fit_type

        # UI elements from the .ui
        self.stdoutEdit.setReadOnly(True)
        self.cancelButton.clicked.connect(self.cancel)
        self.closeButton.clicked.connect(self.close)

        # Worker thread
        self.worker = FitWorker(path, band, solver, max_threads, fit_type)
        self.worker.output.connect(self._append_output)
        self.worker.finished.connect(self._finished)

        self.titleLabel.setText(f"IMFIT: {os.path.basename(self.path)}  band={self.band}  solver={self.solver}")
        self.statusLabel.setText("Status: Running")

        # Start
        self.worker.start()

    def _append_output(self, text):
        # Append text to stdout view
        self.stdoutEdit.moveCursor(QTextCursor.MoveOperation.End)
        self.stdoutEdit.insertPlainText(text)
        self.stdoutEdit.moveCursor(QTextCursor.MoveOperation.End)

    def _finished(self, code):
        self.statusLabel.setText(f"Status: Finished (code={code})")
        # If parent is the main window, trigger a refresh of the currently selected galaxy
        try:
            parent = self.parent()
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
            self.statusLabel.setText("Status: Terminated")
        except Exception as e:
            self.stdoutEdit.insertPlainText(f"Failed to terminate: {e}\n")

