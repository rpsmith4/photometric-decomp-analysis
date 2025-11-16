from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import *
from PyQt6 import uic
import os
from pathlib import Path
import subprocess
import sys
import glob
import argparse
import shutil
sys.path.append("../")
import imfit_run


class MainWindow(QDialog):
    def __init__(self, galpathlist=None):
        super().__init__()
        self.ui = uic.loadUi('ds9_open.ui', self)


        self.galpathlist = galpathlist
        self.curr_gal_index = 0
        self.solvertype = "LM"
        self.band = "g"

        self.ui.LMbutton.clicked.connect(lambda: self.set_solver("LM"))
        self.ui.NMbutton.clicked.connect(lambda: self.set_solver("NM"))
        self.ui.DEbutton.clicked.connect(lambda: self.set_solver("DE"))
        self.ui.gbutton.clicked.connect(lambda: self.set_band("g"))
        self.ui.rbutton.clicked.connect(lambda: self.set_band("r"))
        self.ui.ibutton.clicked.connect(lambda: self.set_band("i"))
        self.ui.zbutton.clicked.connect(lambda: self.set_band("z"))
        self.ui.markfitted.clicked.connect(lambda: self.markgalaxy("fitted"))
        self.ui.markreturn.clicked.connect(lambda: self.markgalaxy("return"))
        self.ui.markunable.clicked.connect(lambda: self.markgalaxy("unable"))
        self.ui.saveconfigbutton.clicked.connect(self.saveconfig)

        self.ui.opends9button.clicked.connect(self.open_ds9)
        self.ui.nextgalbutton.clicked.connect(self.next_galaxy)
        self.ui.refitbutton.clicked.connect(self.refit)
        self.ui.cancelbutton.clicked.connect(self.cancel)

        self.currentgalaxytext = self.ui.currentgalaxytext
        self.currentgalaxytext.setText(f"Current Galaxy: {os.path.basename(galpathlist[self.curr_gal_index])}")
        self.currentgalaxytext.repaint()

        l = self.ui.label
        p = self.galpathlist[self.curr_gal_index]
        pixmap = QPixmap(os.path.join(p, "image.jpg"))
        l.setPixmap(pixmap)

        self.galaxylist = self.ui.galaxylist
        self.galaxylist.addItems([os.path.basename(g) for g in galpathlist])
        self.galaxylist.itemSelectionChanged.connect(self.changegal)

        self.colors = {
            "fitted" : "#35bd49",
            "return" : "#bdba35",
            "unable" : "#bd3535"
        }

        self.config = self.ui.configeditor
        self.config.setPlainText("")

        self.params = self.ui.plainTextEdit_2
        self.params.setPlainText("")

        self.ps = []

        self.show()

    def open_ds9(self, Dialog):
        print("Opening DS9...")
        p = self.galpathlist[self.curr_gal_index]

        # files = [os.path.join(p, f"image_{b}.fits") for b in "griz"]
        files = [f"{os.path.join(p, f'2_sersic_{self.band}_composed.fits')}"]
        arg = ["ds9", "-cmap", "inferno", "-scale", "log", "-scale", "limits", "0", "10", "-cube", "3"]
        arg.extend(files)
        subprocess.Popen(arg)
    
    def next_galaxy(self):
        self.curr_gal_index += 1
        self.currentgalaxytext.setText(f"Current Galaxy: {os.path.basename(self.galpathlist[self.curr_gal_index])}")
        self.currentgalaxytext.repaint()

        p = self.galpathlist[self.curr_gal_index]
        with open(os.path.join(p, f"2_sersic_{self.band}.dat"), "r") as f:
            config_file = f.readlines()
        self.config.setPlainText("".join(config_file))
        self.config.repaint()

        with open(os.path.join(p, f"2_sersic_{self.band}_fit_params.txt"), "r") as f:
            config_file = f.readlines()
        self.params.setPlainText("".join(config_file))
        self.params.repaint()
    
    def changegal(self):
        galaxy = self.galaxylist.selectedItems()[0].text()
        gl = [os.path.basename(g) for g in self.galpathlist]
        self.curr_gal_index = gl.index(galaxy)
        self.currentgalaxytext.setText(f"Current Galaxy: {os.path.basename(self.galpathlist[self.curr_gal_index])}")
        self.currentgalaxytext.repaint()

        p = self.galpathlist[self.curr_gal_index]
        with open(os.path.join(p, f"2_sersic_{self.band}.dat"), "r") as f:
            config_file = f.readlines()
        self.config.setPlainText("".join(config_file))
        self.config.repaint()

        with open(os.path.join(p, f"2_sersic_{self.band}_fit_params.txt"), "r") as f:
            config_file = f.readlines()
        self.params.setPlainText("".join(config_file))
        self.params.repaint()

        l = self.ui.label
        pixmap = QPixmap(os.path.join(p, "image.jpg"))
        l.setPixmap(pixmap)
         
    def set_solver(self, solver):
        self.solvertype = solver 

    def set_band(self, band):
        self.band = band
    

    def refit(self):
        imfit_run.main(self.galpathlist[self.curr_gal_index], [self.band], r=False, overwrite=True, mask=True, psf=True, invvar=True, alg="LM", max_threads=8, fit_type="2_sersic", make_composed=True)

        # Just refreshing the configs and stats and whatnot
        self.changegal()
    
    def cancel(self):
        if len(self.ps) > 0:
            self.ps[-1].kill()
            self.ps.pop()
            os.chdir(self.curr_dir)
        else:
            print("No running IMFIT processes")
    
    def markgalaxy(self, markas):
        self.galaxylist.item(self.curr_gal_index).setBackground(QColor(self.colors[markas]))
        self.galaxylist.repaint()
    
    def saveconfig(self):
        p = self.galpathlist[self.curr_gal_index]
        new_config = self.ui.configeditor.toPlainText()
        print(new_config)
        if not(new_config == ""):
            shutil.copyfile(src=os.path.join(p, f"2_sersic_{self.band}.dat"), dst=os.path.join(p, f"2_sersic_{self.band}.dat.bak"))
            with open(os.path.join(p, f"2_sersic_{self.band}.dat"), "w") as f:
                f.write(new_config)


def get_galaxies(p):
    structure = os.walk(p)

    gal_pathlist = []
    for root, dirs, files in structure:
        if not(files == []):
            # Assumes data is at the end of the file tree
            galpath = root
            if galpath != None:
                 gal_pathlist.append(galpath)

    return gal_pathlist

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
