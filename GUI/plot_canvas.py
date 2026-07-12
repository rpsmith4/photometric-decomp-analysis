from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage, QBrush
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import *
import os
from pathlib import Path
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from astropy.visualization.stretch import LogStretch, LinearStretch
from astropy.visualization import ImageNormalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogLocator
import astropy.units as u
import pandas as pd

class PlotCanvas(FigureCanvas):
    def __init__(self, parent = None):
        self.fig = Figure(figsize=(250/50, 250/50), dpi=50)
        self.ax = self.fig.subplots()
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, im, limits, cmap, stretch=LogStretch(), ellipse_params=pd.DataFrame, plottext=None, cbar=True):
        self.fig.clear()
        self.ax = self.fig.subplots()
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        # self.ax.set_axis_off()

        if im.any():
            if limits != None:
                norm = ImageNormalize(stretch=stretch, vmin=limits[0], vmax=limits[1])
                implt = self.ax.imshow(im, origin="lower", norm=norm, cmap=cmap)
            else:
                implt = self.ax.imshow(im, origin="lower", cmap=cmap)

            if cbar:
                # cbbox = inset_axes(self.ax, '15%', '90%', loc = 7)
                self.cbbox = inset_axes(self.ax, '100%', '10%', loc = "lower center", borderpad=-0.3)
                self.cbbox.tick_params(
                    axis = 'both',
                    left = False,
                    top = False,
                    right = False,
                    bottom = False,
                    labelleft = False,
                    labeltop = False,
                    labelright = False,
                    labelbottom = False
                )
                [self.cbbox.spines[k].set_visible(False) for k in self.cbbox.spines]
                self.cbbox.set_facecolor([1,1,1,0.7])

                # cbaxes = inset_axes(cbbox, '30%', '95%', loc = 6)
                cbaxes = inset_axes(self.cbbox, '90%', '30%', loc = "upper center")
                # cb = self.fig.colorbar(implt, cax=cbaxes, orientation="horizontal")#,shrink=0.7,pad=-0.3)
                if isinstance(stretch, LogStretch):
                    cb = self.fig.colorbar(implt, cax=cbaxes, orientation="horizontal", ticks = LogLocator(base=10))
                else:
                    cb = self.fig.colorbar(implt, cax=cbaxes, orientation="horizontal")#,shrink=0.7,pad=-0.3)

                cb.ax.minorticks_on()
                
                # cb.ax.tick_params(labelsize=15) 

            if plottext != None:
                self.ax.text(0.05, 0.95, plottext, size=15,
                        ha="left", va="center",
                        bbox=dict(boxstyle="square",
                                ec=(1., 1, 1),
                                fc=(0,0,0),
                                ),
                        transform=self.ax.transAxes,
                        color="lightgreen"
                        )


            if not ellipse_params.empty:
                host = ellipse_params[ellipse_params["PolarOrHost"] == "Host"].iloc[0]
                polar = ellipse_params[ellipse_params["PolarOrHost"] == "Polar"].iloc[0]
                imshape = im.shape
                ell_host = matplotlib.patches.Ellipse(
                    xy=(imshape[0]/2, imshape[1]/2),
                    height=float(host["semi_minor"]*2),
                    width=float(host["semi_major"]*2),
                    angle=host["angle"],
                    label="Host",
                    ls="--",
                    lw=2,
                    color="red",
                    fill=False
                )
                ell_polar = matplotlib.patches.Ellipse(
                    xy=(imshape[0]/2, imshape[1]/2),
                    height=float(polar["semi_minor"]*2),
                    width=float(polar["semi_major"]*2),
                    angle=polar["angle"],
                    label="Polar",
                    ls="--",
                    lw=2,
                    color="blue",
                    fill=False
                )
                self.ax.add_patch(ell_host)
                self.ax.add_patch(ell_polar)
                self.ax.legend()
        else:
            self.ax.text(0,0.5,"Cannot find FITs image!")

        self.draw()

    def plot_profiles(self, host_data, polar_data, title, overplot=None, y_label='Surface Brightness (mag/arcsec^2)', surfbright=False, is_resid=False, d_A=None):
        self.fig.clear()
        self.ax = self.fig.subplots()

        if d_A == None:
            self.fig.subplots_adjust(left=0.13, right=0.9,bottom=0.1,top=0.90)
        else:
            self.fig.subplots_adjust(left=0.13, right=0.9,bottom=0.1,top=0.85)
        self.ax.cla()
        self.ax.set_title(title, fontsize=12)
        try:
            self.ax.plot(host_data['r'], host_data['mu'], label='Host', color='blue')
            self.ax.plot(polar_data['r'], polar_data['mu'], label='Polar', color='red')
            if overplot:
                self.ax.plot(overplot['host']['r'], overplot['host']['mu'], 'b--', label='Host Image')
                self.ax.plot(overplot['polar']['r'], overplot['polar']['mu'], 'r--', label='Polar Image')
            self.ax.legend(fontsize=8)
            self.ax.set_xlabel('Radius (arcsec)', fontsize=10)
            self.ax.set_ylabel(y_label, fontsize=10)
            self.ax.tick_params(labelsize=8)
            self.ax.grid()
            self.ax.minorticks_on()
            if surfbright:
                self.ax.invert_yaxis()

            if is_resid:
                self.ax.axhline(y=0, ls="--", color="black")

            if d_A != None:
                self.secax = self.ax.secondary_xaxis('top', functions=(
                    lambda x: ((x*u.arcsec * d_A).to(u.kpc, u.dimensionless_angles())).value,
                    lambda y: ((y*u.kpc / d_A).to(u.arcsec, u.dimensionless_angles())).value
                    ))
                self.secax.set_xlabel('Radius (kpc)', fontsize=10)
        except:
            self.ax.text(0.2,0.5, "No Radial Data!", fontsize=24)
        self.draw()


