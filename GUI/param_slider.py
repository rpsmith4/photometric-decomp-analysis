from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage, QBrush
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import *
import os
from pathlib import Path
import astropy.units as u
import math

from utils import clearLayout
from utils import DataSet

class ConfigAdjustWidget(QWidget):
    def __init__(self, parent, dataset: DataSet):
        self.layout = parent
        self.dataset = dataset
        self.config_dict = self.dataset.config_dict
        self.function_list = self.config_dict["function_sets"][0]["function_list"]
        self.param_widgets = {}
    
    def draw_config_adjust(self):
            self.param_widgets = {}
            for func_idx, func in enumerate(self.function_list):
                params = func["parameters"]
                label = func["label"]

                h = QHBoxLayout()
                comp_sel = QCheckBox()
                comp_sel.setChecked(True)
                comp_sel.setFixedWidth(5)
                comp_sel.stateChanged.connect(
                    lambda state, func_idx=func_idx: self.on_component_checkbox_changed(func_idx, state)
                )

                label_text = QTextBrowser()
                label_text.setText(label)
                label_text.setMaximumHeight(30)
                label_text.setMinimumWidth(50)
                label_text.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
                label_text.setAlignment(QtCore.Qt.AlignCenter)

                h.addWidget(comp_sel)
                h.addWidget(label_text)
                self.layout.addLayout(h)

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
                    self.draw_params(initval, lowlim, hilim, fixed, (func_idx, param), label, self.layout)

    def draw_params(self, initval, lowlim, hilim, fixed, paramkey, label, layout):
        func_idx, paramname = paramkey
        ndigits = 6
        widget = ParamSliderWidget(paramname, initval, lowlim, hilim, fixed=fixed, ndigits=ndigits, d_A=self.dataset.d_A)
        widget.setMinimumWidth(100)
        layout.addWidget(widget)
        self.param_widgets[paramkey] = widget

    def on_component_checkbox_changed(self, func_idx, state): # Change the config file to a new one with only the selected parameters, also change the composed image and the mask
        # Have to somehow get the information from the mainwindow
        print("Hello")
        print(state)

class ParamSliderWidget(QWidget):
    def __init__(self, paramname, initval, lowlim, hilim, fixed=False, ndigits=3, parent=None, d_A=None):
        super().__init__(parent)
        self.paramname = paramname
        self.ndigits = ndigits
        # Slider will use a fixed integer range and we'll map it
        # linearly to the actual parameter range [min, max].
        self._slider_steps = 10000
        self.fixed = fixed
        self.d_A = d_A # Angular size distance (I should probably find a way to avoid just passing it to here but whatever)

        parameter_adjust_layout = QHBoxLayout()
        parameter_adjust_layout.setContentsMargins(0,0,0,0)

        self.text = QTextBrowser()
        self.text.setText(str(paramname))
        self.text.setStyleSheet('font-size: 10px')
        self.text.setMaximumSize(45,25)
        self.text.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,QtWidgets.QSizePolicy.Policy.Maximum)
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.fixed_checkbox = QCheckBox("Fixed")
        self.fixed_checkbox.setChecked(fixed)

        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,QtWidgets.QSizePolicy.Policy.Maximum)
        # self.slider.setTickInterval(5)
        self.slider.setSingleStep(1)
        # Use fixed slider range and map to [lowlim, hilim]
        self.slider.setRange(0, self._slider_steps)
        try:
            frac = 0.0 if hilim == lowlim else (initval - lowlim) / float(hilim - lowlim)
        except Exception:
            frac = 0.0
        self.slider.setValue(int(round(max(0.0, min(1.0, frac)) * self._slider_steps)))

        parameter_adjust_layout.addWidget(self.text)
        parameter_adjust_layout.addWidget(self.slider)
        parameter_adjust_layout.setStretchFactor(self.slider, 4)
        parameter_adjust_layout.addWidget(self.fixed_checkbox)

        spinboxes_layout = QHBoxLayout()
        from scientific_spinbox import ScientificDoubleSpinBox
        self.minspinbox = ScientificDoubleSpinBox()
        # self.minspinbox.setDecimals(ndigits)
        spinbox_minwidth = 50
        self.minspinbox.setMaximum(hilim)
        self.minspinbox.setMinimum(-1e9)
        self.minspinbox.setValue(lowlim)
        self.minspinbox.setMaximumWidth(100)
        self.minspinbox.setMinimumWidth(spinbox_minwidth)
        self.minspinbox.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

        self.valspinbox = ScientificDoubleSpinBox()
        # self.valspinbox.setDecimals(ndigits)
        self.valspinbox.setMaximum(hilim)
        self.valspinbox.setMinimum(lowlim)
        self.valspinbox.setValue(initval)
        self.valspinbox.setMaximumWidth(100)
        self.valspinbox.setMinimumWidth(spinbox_minwidth)
        self.valspinbox.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

        self.maxspinbox = ScientificDoubleSpinBox()
        # self.maxspinbox.setDecimals(ndigits)
        self.maxspinbox.setSingleStep(1e-2)
        self.maxspinbox.setMinimum(lowlim)
        self.maxspinbox.setMaximum(1e9)
        self.maxspinbox.setValue(hilim)
        self.maxspinbox.setMaximumWidth(100)
        self.maxspinbox.setMinimumWidth(spinbox_minwidth)
        self.maxspinbox.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        # self.maxspinbox.setFont("arial: size=20px")

        spinboxes_layout.addWidget(self.minspinbox)
        spinboxes_layout.addWidget(self.valspinbox)
        spinboxes_layout.addWidget(self.maxspinbox)

        spinboxes_layout.setStretchFactor(self.minspinbox, 1)
        spinboxes_layout.setStretchFactor(self.valspinbox, 1)
        spinboxes_layout.setStretchFactor(self.maxspinbox, 1)
        parameter_adjust_layout.addLayout(spinboxes_layout)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.converted_label = QLabel()
        self.converted_label.setStyleSheet('font-size: 10px')
        self.converted_label.setMaximumHeight(12)
        layout.addWidget(self.converted_label)
        layout.addLayout(parameter_adjust_layout)
        self.setLayout(layout)

        self.set_fixed_state(fixed)

        self.slider.valueChanged.connect(self.slider_changed)
        self.valspinbox.setKeyboardTracking(False)
        self.valspinbox.valueChanged.connect(self.spinbox_changed)
        self.minspinbox.setKeyboardTracking(False)
        self.minspinbox.valueChanged.connect(self.minspinbox_changed)
        self.maxspinbox.setKeyboardTracking(False)
        self.maxspinbox.valueChanged.connect(self.maxspinbox_changed)
        self.fixed_checkbox.stateChanged.connect(lambda state: self.set_fixed_state(state==2))
        
        spinboxes_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        parameter_adjust_layout.setStretchFactor(spinboxes_layout, 10)
        
        self.update_converted()

    def set_fixed_state(self, is_fixed):
        self.minspinbox.setEnabled(not is_fixed)
        self.maxspinbox.setEnabled(not is_fixed)
        # self.slider.setEnabled(not is_fixed)
        # self.valspinbox.setEnabled(not is_fixed)

    def slider_changed(self, value):
        # Map slider integer (0.._slider_steps) linearly to [min, max]
        minval = self.minspinbox.value()
        maxval = self.maxspinbox.value()
        if maxval == minval:
            float_val = minval
        else:
            float_val = minval + (value / float(self._slider_steps)) * (maxval - minval)
        self.valspinbox.blockSignals(True)
        self.valspinbox.setValue(float_val)
        self.valspinbox.blockSignals(False)
        self.update_converted()

    def spinbox_changed(self, value):
        # Map the spinbox value into the slider integer range
        minval = self.minspinbox.value()
        maxval = self.maxspinbox.value()
        if maxval == minval:
            int_val = 0
        else:
            frac = (value - minval) / float(maxval - minval)
            int_val = int(round(max(0.0, min(1.0, frac)) * self._slider_steps))
        self.slider.blockSignals(True)
        self.slider.setValue(int_val)
        self.slider.blockSignals(False)
        self.update_converted()

    def minspinbox_changed(self, new_min):
        cur_max = self.maxspinbox.value()
        if new_min > cur_max:
            new_min = cur_max
            self.maxspinbox.setValue(cur_max)

        self.valspinbox.setMinimum(new_min)
        self.maxspinbox.setMinimum(new_min)
        if self.valspinbox.value() < new_min:
            self.valspinbox.setValue(new_min)
        # Recompute slider position to respect new bounds
        cur_val = self.valspinbox.value()
        self.spinbox_changed(cur_val)

    def maxspinbox_changed(self, new_max):
        cur_min = self.minspinbox.value()
        if new_max < cur_min:
            cur_min = new_max 
            self.minspinbox.setValue(new_max)

        self.valspinbox.setMaximum(new_max)
        self.minspinbox.setMaximum(new_max)
        if self.valspinbox.value() > new_max:
            self.valspinbox.setValue(new_max)
        # Recompute slider position to respect new bounds
        cur_val = self.valspinbox.value()
        self.spinbox_changed(cur_val)

    def get_values(self):
        return {
            'value': self.valspinbox.value(),
            'min': self.minspinbox.value(),
            'max': self.maxspinbox.value(),
            'fixed': self.fixed_checkbox.isChecked()
        }

    def update_converted(self):
        val = self.valspinbox.value()
        minval = self.minspinbox.value()
        maxval = self.maxspinbox.value()
        if self.paramname in ["r_e", "R_ring", "sigma_r"]:
            valarcsec = val * 0.262 * u.arcsec
            minvalarcsec = minval * 0.262 * u.arcsec
            maxvalarcsec = maxval * 0.262 * u.arcsec

            if self.d_A != None:
                minvalkpc = (minvalarcsec * self.d_A).to(u.kpc, u.dimensionless_angles())
                maxvalkpc = (maxvalarcsec * self.d_A).to(u.kpc, u.dimensionless_angles())
                valkpc = (valarcsec * self.d_A).to(u.kpc, u.dimensionless_angles())
                self.converted_label.setText(f"Min: {minvalarcsec.value:.3f} Val: {valarcsec.value:.3f} Max: {maxvalarcsec.value:.3f} arcsec (Min: {minvalkpc.value:.3f} Val: {valkpc.value:.3f} Max: {maxvalkpc.value:.3f} kpc)")
            else:
                self.converted_label.setText(f"Min: {minvalarcsec.value:.3f} Val: {valarcsec.value:.3f} Max: {maxvalarcsec.value:.3f} arcsec")
        elif self.paramname in ["I_e", "A"]:
            if val > 0 and minval > 0 and maxval > 0:
                valmag = 22.5 - 2.5 * math.log10(val/0.262**2)
                minvalmag = 22.5 - 2.5 * math.log10(minval/0.262**2)
                maxvalmag = 22.5 - 2.5 * math.log10(maxval/0.262**2)
                self.converted_label.setText(f"Min: {minvalmag:.3f} Val: {valmag:.3f} Max: {maxvalmag:.3f} mag/arcsec^2")
            else:
                self.converted_label.setText("N/A")
        else:
            self.converted_label.setText("")
