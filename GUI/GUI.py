#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:00:06 2021

@author: robgc
"""
from multiprocessing import set_start_method
import csv
import sys
import numpy as np
from scipy.constants import c
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from backend import tcalc, pop_calc, FWHM, contrast, ncalc
from vals_413 import d12_413, func_d23_413, func_spon_413, spontaneous_21_413, kp_413, \
                        func_Ic_413, func_Ip_413, func_omega_c_413, func_omega_p_413, func_kc_413
from vals_318 import d12_318, func_d23_318, kp_318, func_spon_318, spontaneous_21_318,\
                        func_Ic_318, func_Ip_318, func_omega_c_318, func_omega_p_318, func_kc_318
import matplotlib.pyplot as plt


class UI(QMainWindow):

    def __init__(self, *args, **kwargs):
        """
        Initialisation of the GUI window and all of its
        features
        """
        self.spontaneous_21_413 = spontaneous_21_413
        self.spontaneous_21_318 = spontaneous_21_318
        super(UI, self).__init__(*args, **kwargs)
        self.setWindowTitle("Sr88 3 Level System Simulator - By Robert Noakes")
        self.screen = QDesktopWidget().screenGeometry(-1)
        self.setFixedSize(640, 700)
        self.overallLayout = QHBoxLayout()
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()
        self.overallLayout.addLayout(self.leftLayout)
        self.overallLayout.addLayout(self.rightLayout)
        self.overallLayout.setSpacing(10)
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.overallLayout)
        self.add_toolbar()
        self.add_dropdowns()
        self.add_inputs()
        self.add_checkboxes()
        self.add_images()
    
    def add_toolbar(self):
        """
        Defines buttons on the bottom toolbar and 
        their links to different functions
        """
        self.toolbar = QToolBar()
        self.addToolBar(Qt.BottomToolBarArea, self.toolbar)
        self.load_button = QAction("Load CSV", self)
        self.load_button.setStatusTip("Load CSV")
        self.load_button.triggered.connect(self.load_csv)
        self.toolbar.addAction(self.load_button)
        self.save_button = QAction("Save CSV", self)
        self.save_button.setStatusTip("Save CSV")
        self.save_button.triggered.connect(self.save_csv)
        self.toolbar.addAction(self.save_button)
        self.clear_button = QAction("Clear", self)
        self.clear_button.setStatusTip("Clear")
        self.toolbar.addAction(self.clear_button)
        self.clear_button.triggered.connect(self.clear_text)
        self.plot_button = QAction("Plot Transmission", self)
        self.plot_button.setStatusTip("Plot Transmission")
        self.toolbar.addAction(self.plot_button)
        self.plot_button.triggered.connect(self.transmission)
        self.pop_button = QAction("Plot Population", self)
        self.pop_button.setStatusTip("Plot Population")
        self.toolbar.addAction(self.pop_button)
        self.pop_button.triggered.connect(self.population)
        self.sl_button = QAction("Plot Group Index", self)
        self.sl_button.setStatusTip("Plot Group Index")
        self.toolbar.addAction(self.sl_button)
        self.sl_button.triggered.connect(self.slowlight)
        self.exit_button = QAction("Exit", self)
        self.exit_button.setStatusTip("Exit")
        self.toolbar.addAction(self.exit_button)
        self.exit_button.triggered.connect(self.close)
        
    def add_images(self):
        """
        Adds the images to the RHS of the window
        """
        self.leveldiagram = QLabel()
        self.levelpixmap = QPixmap('imgs/orig.png')
        self.leveldiagram.setPixmap(self.levelpixmap)
        self.rightLayout.addWidget(self.leveldiagram, alignment=Qt.AlignHCenter | Qt.AlignTop)
        self.Rabi_label = QLabel()
        self.Rabi_pixmap = QPixmap('imgs/Rabi.png')
        self.Rabi_label.setPixmap(self.Rabi_pixmap)
        self.rightLayout.addWidget(self.Rabi_label, alignment=Qt.AlignHCenter | Qt.AlignBottom)
        self.pow_label = QLabel()
        self.pow_pixmap = QPixmap('imgs/intensity.png')
        self.pow_label.setPixmap(self.pow_pixmap)
        self.rightLayout.addWidget(self.pow_label, alignment=Qt.AlignHCenter | Qt.AlignCenter)
    
    def add_dropdowns(self):
        """
        Adds the dropdown selectors for choosing the Rydberg series and
        input method for laser parameters
        """
        self.system_choice = QComboBox()
        self.system_choice.addItems(["Sr88 1D2 Series", "Sr88 3D2 Series"])
        self.system_choice.setCurrentIndex(0)
        self.leftLayout.addWidget(self.system_choice)
        self.input_type = QComboBox()
        self.input_type.addItems(["Enter Laser Powers and Diameters", "Enter Laser Intensities", "Enter Rabi Frequencies"])
        self.input_type.currentIndexChanged.connect(self.grey)
        self.input_type.setCurrentIndex(0)
        self.leftLayout.addWidget(self.input_type)
        
        
    def add_checkboxes(self):
        """
        Adds the checkboxes for Doppler and Transit-time broadening
        """
        self.doppler = QCheckBox("Include Doppler Broadening?")
        self.transit = QCheckBox ("Include Transit Time Broadening?")
        self.leftLayout.addWidget(self.doppler)
        self.leftLayout.addWidget(self.transit)
        
    def add_inputs(self):
        """
        Adds the labels and text boxes for entering the model parameters
        """
        self.labels = {}
        self.boxes = {}
        inputs_layout = QGridLayout()
        
        labels = {"Rydberg State n level" : (0, 0),
                  "Probe Laser Power (W)" : (1, 0), 
                  "Coupling Laser Power (W)" : (2, 0),
                  "Probe Laser Diameter (m)" : (3, 0),
                  "Coupling Laser Diameter (m)" : (4, 0),
                  "Probe Laser Intensity (W/m^2)" : (5, 0), 
                  "Coupling Laser Intensity (W/m^2)" : (6, 0), 
                  "Probe Rabi Frequency (Hz)" : (7, 0), 
                  "Coupling Rabi Frequency (Hz)" : (8, 0), 
                  "Probe Laser Linewidth (Hz)" : (9, 0), 
                  "Coupling Laser Linewidth (Hz)" : (10, 0),
                  "Atomic Density (m^-3)" : (11, 0),
                  "Atomic Beam Width (m)" : (12, 0), 
                  "Oven Temperature (K)" : (13, 0), 
                  "Atomic Beam Divergence Angle (Rad)" : (14, 0),
                  "Minimum Detuning (Hz)" : (15, 0),
                  "Maximum Detunung (Hz)" : (16, 0),
                  "Number of Plotted Points" : (17, 0),
                  "Coupling Laser Detuning (Hz)" : (18, 0)}
        
        boxes = {"n" : (0, 1),
                 "pp" : (1, 1), 
                 "cp" : (2, 1), 
                 "pd" : (3, 1),
                 "cd" : (4, 1),
                 "Ip" : (5, 1), 
                 "Ic" : (6, 1), 
                 "omega_p" : (7, 1), 
                 "omega_c" : (8, 1),
                 "lwp" : (9, 1),
                 "lwc" : (10, 1),
                 "density" : (11, 1), 
                 "sl" : (12, 1), 
                 "T" : (13, 1), 
                 "alpha" : (14, 1),
                 "dmin" : (15, 1),
                 "dmax" : (16, 1),
                 "steps" : (17, 1),
                 "delta_c" : (18, 1)}
        
        for text, pos in labels.items():
            self.labels[text] = QLabel(text)
            inputs_layout.addWidget(self.labels[text], pos[0], pos[1])
        
        for text, pos in boxes.items():
            self.boxes[text] = QLineEdit()
            inputs_layout.addWidget(self.boxes[text], pos[0], pos[1])
        
        laser_params = ["Ip", "Ic", "omega_p", "omega_c"]
        for i in laser_params:
            self.boxes[i].setReadOnly(True)

        self.leftLayout.addLayout(inputs_layout)

    def get_text(self, parameter):
        """
        Retrieves the value in the box for a given model parameter
        """
        return self.boxes[parameter].text()
    
    def set_text (self, parameter, text):
        """
        Overwrites the value in the box for a given model parameter
        """
        self.boxes[parameter].setText(text)

    def clear_text(self):
        """
        Clears the value of a given box
        """
        for parameter, val in self.boxes.items():
            self.set_text(parameter, "")
        
    def get_params(self):
        """
        Retrieves the model parameters from the boxes and stores
        them in a dictionary with keyword arguments
        """
        sim_vals = {}
        for param, box in self.boxes.items():
            sim_vals[param] = self.get_text(param)
        return sim_vals
        
    def transmission(self):
        """
        Calculates the probe transmission spectrum
        for a given set of entered model parameters
        """
        vals = self.get_params()
        for param, val in vals.items():
            if vals[param] == "0":
                vals[param] = 0
            if vals[param] == "":
                vals[param] = 0
            else:
                vals[param] = float(val)
        vals["dmin"] = int(vals["dmin"])
        vals["dmax"] = int(vals["dmax"])
        vals["steps"] = int(vals["steps"])
        vals["n"] = int(vals["n"])
        self.vals = vals
        
        if self.system_choice.currentIndex() == 0:
            d23_413 = func_d23_413(vals["n"], "1D2")
            kc_413 = func_kc_413(vals["n"], "1D2")
            if self.input_type.currentIndex() == 0:
                if d23_413 == 0:
                    self.dme()
                    return
                if vals["pp"] == 0 or vals["cp"] == 0 or vals["pd"] == 0 or vals["cd"] == 0:
                    self.power_warn()
                    return
                vals["Ip"] = func_Ip_413(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_413(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"], d23_413)
            if self.input_type.currentIndex() == 1:
                if d23_413 == 0:
                    self.dme()
                    return
                if vals["Ip"] == 0 or vals["Ic"] == 0:
                    self.intensity_warn()
                    return
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"], d23_413)
            if self.input_type.currentIndex() == 2:
                if vals["omega_p"] == 0 or vals["omega_c"] == 0:
                    self.rabi_warn()
                    return
        
        if self.system_choice.currentIndex() == 1:
            d23_318 = func_d23_318(vals["n"], "3D3")
            kc_318 = func_kc_318(vals["n"], "3D3")
            print(kc_318)
            if self.input_type.currentIndex() == 0:
                if d23_318 == 0:
                    self.dme()
                    return
                if vals["pp"] == 0 or vals["cp"] == 0 or vals["pd"] == 0 or vals["cd"] == 0:
                    self.power_warn()
                    return
                vals["Ip"] = func_Ip_318(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_318(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"], d23_318)
            if self.input_type.currentIndex() == 1:
                if d23_318 == 0:
                    self.dme()
                    return
                if vals["Ip"] == 0 or vals["Ic"] == 0:
                    self.intensity_warn()
                    return
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"], d23_318)
            if self.input_type.currentIndex() == 2:
                if vals["omega_p"] == 0 or vals["omega_c"] == 0:
                    self.rabi_warn()
                    return        
        if self.doppler.isChecked():
            gauss = "Y"
        else:
            gauss = "N"

        if self.transit.isChecked():
            if vals["pd"] == 0:
                self.transit_warn()
                return
            if vals["cd"] == 0:
                self.transit_warn()
                return
            tt = "Y"
        else:
            tt = "N"
        
        if self.system_choice.currentIndex() == 0:
            self.spontaneous_32_413 = func_spon_413(vals["n"], "1D2")
            dlist, tlist = tcalc(vals["delta_c"], vals["omega_p"], vals["omega_c"], self.spontaneous_32_413, self.spontaneous_21_413, 
                       vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], gauss, kp_413, kc_413, 
                       vals["density"], d12_413, vals["sl"], vals["T"], vals["alpha"], vals["pd"], vals["cd"], tt)
            self.t_plotter(dlist, tlist)
            
        if self.system_choice.currentIndex() == 1:
            self.spontaneous_32_318 = func_spon_318(vals["n"], "3D3")
            dlist, tlist = tcalc(vals["delta_c"], vals["omega_p"], vals["omega_c"], self.spontaneous_32_318, self.spontaneous_21_318, 
                       vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], gauss, kp_318, kc_318, 
                       vals["density"], d12_318, vals["sl"], vals["T"], vals["alpha"], vals["pd"], vals["cd"], tt)
            self.t_plotter(dlist, tlist)
        
    def population(self):
        """
        Calculates the steady state population spectrum of any state
        for a given set of model parameters
        """
        vals = self.get_params()
        for param, val in vals.items():
            if vals[param] == "":
                vals[param] = 0
            else:
                vals[param] = float(val)
        vals["dmin"] = int(vals["dmin"])
        vals["dmax"] = int(vals["dmax"])
        vals["steps"] = int(vals["steps"])
        vals["n"] = int(vals["n"])
        self.vals = vals
        
        if self.system_choice.currentIndex() == 0:
            d23_413 = func_d23_413(vals["n"], "1D2")
            kc_413 = func_kc_413(vals["n"], "1D2")
            if self.input_type.currentIndex() == 0:
                if d23_413 == 0:
                    self.dme()
                    return
                if vals["pp"] == 0 or vals["cp"] == 0 or vals["pd"] == 0 or vals["cd"] == 0:
                    self.power_warn()
                    return
                vals["Ip"] = func_Ip_413(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_413(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"], d23_413)
            if self.input_type.currentIndex() == 1:
                if d23_413 == 0:
                    self.dme()
                    return
                if vals["Ip"] == 0 or vals["Ic"] == 0:
                    self.intensity_warn()
                    return
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"], d23_413)
            if self.input_type.currentIndex() == 2:
                if vals["omega_p"] == 0 or vals["omega_c"] == 0:
                    self.rabi_warn()
                    return
        
        if self.system_choice.currentIndex() == 1:
            d23_318 = func_d23_318(vals["n"], "3D3")
            kc_318 = func_kc_318(vals["n"], "3D3")
            if self.input_type.currentIndex() == 0:
                if d23_318 == 0:
                    self.dme()
                    return
                if vals["pp"] == 0 or vals["cp"] == 0 or vals["pd"] == 0 or vals["cd"] == 0:
                    self.power_warn()
                    return
                vals["Ip"] = func_Ip_318(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_318(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"], d23_318)
            if self.input_type.currentIndex() == 1:
                if d23_413 == 0:
                    self.dme()
                    return
                if vals["Ip"] == 0 or vals["Ic"] == 0:
                    self.intensity_warn()
                    return
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"], d23_318)
            if self.input_type.currentIndex() == 2:
                if vals["omega_p"] == 0 or vals["omega_c"] == 0:
                    self.rabi_warn()
                    return
        if self.doppler.isChecked():
            gauss = "Y"
        else:
            gauss = "N"

        if self.transit.isChecked():
            if vals["pd"] == 0:
                self.transit_warn()
                return
            if vals["cd"] == 0:
                self.transit_warn()
                return
            tt = "Y"
        else:
            tt = "N"
        
        self.showdialog()
        try:
            if self.system_choice.currentIndex() == 0:
                self.spontaneous_32_413 = func_spon_413(vals["n"], "1D2")
                dlist, plist = pop_calc(vals["delta_c"], vals["omega_p"], vals["omega_c"], self.spontaneous_32_413, self.spontaneous_21_413, 
                       vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], self.state_index, gauss, 
                       vals["T"], kp_413, kc_413, vals["alpha"], vals["pd"], vals["cd"], tt)
                self.p_plotter(dlist, plist)           
            
            if self.system_choice.currentIndex() == 1:
                self.spontaneous_32_318 = func_spon_318(vals["n"], "3D3")
                dlist, plist = pop_calc(vals["delta_c"], vals["omega_p"], vals["omega_c"], self.spontaneous_32_318, self.spontaneous_21_318, 
                       vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], self.state_index, gauss, 
                       vals["T"], kp_318, kc_318, vals["alpha"], vals["pd"], vals["cd"], tt)
                self.p_plotter(dlist, plist)
        except:
            return


    def slowlight(self):
        """
        Calculates the probe transmission spectrum
        for a given set of entered model parameters
        """
        vals = self.get_params()
        for param, val in vals.items():
            if vals[param] == "0":
                vals[param] = 0
            if vals[param] == "":
                vals[param] = 0
            else:
                vals[param] = float(val)
        vals["dmin"] = int(vals["dmin"])
        vals["dmax"] = int(vals["dmax"])
        vals["steps"] = int(vals["steps"])
        vals["n"] = int(vals["n"])
        self.vals = vals
        
        if self.system_choice.currentIndex() == 0:
            d23_413 = func_d23_413(vals["n"], "1D2")
            kc_413 = func_kc_413(vals["n"], "1D2")
            if self.input_type.currentIndex() == 0:
                if d23_413 == 0:
                    self.dme()
                    return
                if vals["pp"] == 0 or vals["cp"] == 0 or vals["pd"] == 0 or vals["cd"] == 0:
                    self.power_warn()
                    return
                vals["Ip"] = func_Ip_413(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_413(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"], d23_413)
            if self.input_type.currentIndex() == 1:
                if d23_413 == 0:
                    self.dme()
                    return
                if vals["Ip"] == 0 or vals["Ic"] == 0:
                    self.intensity_warn()
                    return
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"], d23_413)
            if self.input_type.currentIndex() == 2:
                if vals["omega_p"] == 0 or vals["omega_c"] == 0:
                    self.rabi_warn()
                    return
        
        if self.system_choice.currentIndex() == 1:
            d23_318 = func_d23_318(vals["n"], "3D3")
            kc_318 = func_kc_318(vals["n"], "3D3")
            print(kc_318)
            if self.input_type.currentIndex() == 0:
                if d23_318 == 0:
                    self.dme()
                    return
                if vals["pp"] == 0 or vals["cp"] == 0 or vals["pd"] == 0 or vals["cd"] == 0:
                    self.power_warn()
                    return
                vals["Ip"] = func_Ip_318(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_318(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"], d23_318)
            if self.input_type.currentIndex() == 1:
                if d23_318 == 0:
                    self.dme()
                    return
                if vals["Ip"] == 0 or vals["Ic"] == 0:
                    self.intensity_warn()
                    return
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"], d23_318)
            if self.input_type.currentIndex() == 2:
                if vals["omega_p"] == 0 or vals["omega_c"] == 0:
                    self.rabi_warn()
                    return        
        if self.doppler.isChecked():
            gauss = "Y"
        else:
            gauss = "N"

        if self.transit.isChecked():
            if vals["pd"] == 0:
                self.transit_warn()
                return
            if vals["cd"] == 0:
                self.transit_warn()
                return
            tt = "Y"
        else:
            tt = "N"
        
        if self.system_choice.currentIndex() == 0:
            self.spontaneous_32_413 = func_spon_413(vals["n"], "1D2")
            dlist, nlist = ncalc(vals["delta_c"], vals["omega_p"], vals["omega_c"], self.spontaneous_32_413, self.spontaneous_21_413, 
                       vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], gauss, kp_413, kc_413, 
                       vals["density"], d12_413, vals["sl"], vals["T"], vals["alpha"], vals["pd"], vals["cd"], tt)
            self.n_plotter(dlist, nlist)
            
        if self.system_choice.currentIndex() == 1:
            self.spontaneous_32_318 = func_spon_318(vals["n"], "3D3")
            dlist, nlist = ncalc(vals["delta_c"], vals["omega_p"], vals["omega_c"], self.spontaneous_32_318, self.spontaneous_21_318, 
                       vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], gauss, kp_318, kc_318, 
                       vals["density"], d12_318, vals["sl"], vals["T"], vals["alpha"], vals["pd"], vals["cd"], tt)
            self.n_plotter(dlist, nlist)
            
    def load_csv(self):
        """
        Allows loading in of previously saved model parameters
        in a .csv file
        """
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setNameFilter("csv (*.csv)")
        selected = dlg.exec()
        if selected:
            self.filename = dlg.selectedFiles()[0]
            dlg.close()
        else:
            dlg.close()
            return
        if self.filename == "":
            dlg.close()
            return
        
        with open(f"{self.filename}", "rt") as file: 
            reader = csv.reader(file, delimiter=',')
            for rows in reader:
                param = rows[0]
                val = rows[1]
                self.set_text(param, val)
            
    def save_csv(self):
        """
        Allows saving of model parameters in a .csv file
        to reload in later
        """
        dlg = QFileDialog()
        self.filename = dlg.getSaveFileName(self, 'Save File')[0]
        if self.filename == "":
            dlg.close()
            return
        if not self.filename.endswith('.csv'):
            self.filename += '.csv'
        with open(f"{self.filename}", "w", newline='', encoding='utf-8') as file:    
            write = csv.writer(file, delimiter=',')    
            for param, val in self.get_params().items():
                write.writerow([param, val])                        
        
    def grey(self, i):
        """
        Limits boxes which are not necesary on the interface to
        read only when they are not needed
        """
        if i == 0:
            self.set_text("Ip", "")
            self.set_text("Ic", "")
            self.set_text("omega_p", "")
            self.set_text("omega_c", "")
            self.boxes["pp"].setReadOnly(False)
            self.boxes["cp"].setReadOnly(False)
            self.boxes["Ip"].setReadOnly(True)
            self.boxes["Ic"].setReadOnly(True)
            self.boxes["omega_p"].setReadOnly(True)
            self.boxes["omega_c"].setReadOnly(True)
        if i == 1:
            self.set_text("pp", "")
            self.set_text("cp", "")
            self.set_text("omega_p", "")
            self.set_text("omega_c", "")
            self.boxes["Ip"].setReadOnly(False)
            self.boxes["Ic"].setReadOnly(False)
            self.boxes["omega_p"].setReadOnly(True)
            self.boxes["omega_c"].setReadOnly(True)
        if i == 2:
            self.set_text("pp", "")
            self.set_text("cp", "")
            self.set_text("Ip", "")
            self.set_text("Ic", "")
            self.boxes["pp"].setReadOnly(True)
            self.boxes["cp"].setReadOnly(True)
            self.boxes["Ip"].setReadOnly(True)
            self.boxes["Ic"].setReadOnly(True)
            self.boxes["omega_p"].setReadOnly(False)
            self.boxes["omega_c"].setReadOnly(False)
            
    def showdialog(self):
        """
        Brings up a dialog box which allows selection of which
        state population to plot
        """
        self.d = QDialog()
        self.dd = QComboBox(self.d)
        self.dd.move(100, 0)
        self.dd.addItems(["Ground", "Intermediate", "Rydberg"])
        self.dd.setCurrentIndex(0)
        self.b1 = QPushButton("ok",self.d)
        self.b1.move(110, 50)
        self.b1.clicked.connect(self.state)
        
        self.d.setWindowTitle("Choose State to Plot")
        self.d.setWindowModality(Qt.ApplicationModal)
        self.d.exec_()
        
    def state(self):
        """
        Converts dialog box value back into choice of state
        that the model accepts
        """
        if self.dd.currentIndex() == 0:
            self.state_number = "Ground"
            self.state_index = 0,0
        if self.dd.currentIndex() == 1:
            self.state_number = "Intermediate"
            self.state_index = 1,1
        if self.dd.currentIndex() == 2:
            self.state_number = "Rydberg"
            self.state_index = 2,2
        self.d.close()
        
    def save_dialog(self, event):
        """
        Allows saving of plot data upon closing the figure
        """
        box = QMessageBox.question(self, 'Save', 'Do you want to save data as a .csv?')
        if box == QMessageBox.Yes:
            dlg = QFileDialog()
            self.filename = dlg.getSaveFileName(self, 'Save File')[0]
            if self.filename == "":
                dlg.close()
                return
            if not self.filename.endswith('.csv'):
                self.filename += '.csv'
            with open(f"{self.filename}", "w", newline='', encoding='utf-8') as file:    
                write = csv.writer(file, delimiter=',')    
                if self.typ == "t":
                    write.writerow(["Probe Detununing", "Probe Transmission"])
                if self.typ == "p":
                    write.writerow(["Probe Detununing", f"{self.state_number} Population"])
                if self.typ == "n":
                    write.writerow(["Probe Detununing", "Group Index"])
                for i in range(len(self.dlist)):
                    write.writerow([self.dlist[i], self.flist[i]])
                    
    
    def t_plotter(self, dlist, tlist):
        """
        Creates the figure and legend for a given array of probe 
        transmission values and probe detunings
        """
        if self.system_choice.currentIndex() == 0:
            self.spontaneous_21 = self.spontaneous_21_413
            self.spontaneous_32 = self.spontaneous_32_413
        if self.system_choice.currentIndex() == 1:
            self.spontaneous_21 = self.spontaneous_21_318
            self.spontaneous_32 = self.spontaneous_32_318
        self.dlist = dlist
        self.flist = tlist
        self.typ = "t"
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        """ Geometric library to calculate linewidth of EIT peak (FWHM) """
        pw = FWHM(dlist, tlist)
        ct = contrast(dlist, tlist)
        if pw != 0 and ct != 0:
            if pw < 2*np.pi*1e6:
                ax.text(0.8, 0.07, f"EIT FWHM = 2$\pi$ x {pw/(1e3*2*np.pi):.2f} $kHz$", transform=ax.transAxes, fontsize=10, va='center', ha='center')
            else:
                ax.text(0.8, 0.07, f"EIT FWHM = 2$\pi$ x {pw/(1e6*2*np.pi):.2f} $MHz$", transform=ax.transAxes, fontsize=10, va='center', ha='center')
            ax.text(0.8, 0.13, f"EIT Contrast = {ct:.3f}", transform=ax.transAxes, fontsize=10, va='center', ha='center')        
        else:
            pw = FWHM(dlist, -tlist)
            if pw < 2*np.pi*1e6:
                ax.text(0.5, 0.97, f"Background FWHM = 2$\pi$ x {pw/(1e3*2*np.pi):.2f} $kHz$", transform=ax.transAxes, fontsize=10, va='center', ha='center')
            else:
                ax.text(0.5, 0.97, f"Background FWHM = 2$\pi$ x {pw/(1e6*2*np.pi):.2f} $MHz$", transform=ax.transAxes, fontsize=10, va='center', ha='center')       
        
        plt.title(r"Probe transmission against probe beam detuning")
        if dlist[-1]-dlist[0] >= 1e6:
            ax.plot(dlist/(1e6), tlist, color="orange", label="$\Omega_c=$" f"{self.vals['omega_c']:.2e} $Hz$"\
                "\n" "$\Omega_p=$" f"{self.vals['omega_p']:.2e} $Hz$" "\n" \
                "$\Gamma_{c}$" f"= {self.spontaneous_32/(2*np.pi):.2e} $Hz$" "\n" \
                "$\Gamma_{p}$" f"= {self.spontaneous_21/(2*np.pi):.2e} $Hz$" "\n"\
                "$\Delta_c =$" f"{self.vals['delta_c']/1e6:.2f} $Hz$" "\n" \
                f"$\gamma_p$ = {self.vals['lwp']:.2e} $Hz$" "\n" 
                f"$\gamma_c$ = {self.vals['lwc']:.2e} $Hz$")
            ax.set_xlabel(r"$\Delta_p$ / MHz")
        else:
            ax.plot(dlist/(1e3), tlist, color="orange", label="$\Omega_c=$" f"{self.vals['omega_c']:.2e} $Hz$"\
                "\n" "$\Omega_p=$" f"{self.vals['omega_p']:.2e} $Hz$" "\n" \
                "$\Gamma_{c}$" f"= {self.spontaneous_32/(2*np.pi):.2e} $Hz$" "\n" \
                "$\Gamma_{p}$" f"= {self.spontaneous_21/(2*np.pi):.2e} $Hz$" "\n"\
                "$\Delta_c =$" f"{self.vals['delta_c']/1e6:.2f} $Hz$" "\n" \
                f"$\gamma_p$ = {self.vals['lwp']:.2e} $Hz$" "\n" 
                f"$\gamma_c$ = {self.vals['lwc']:.2e} $Hz$")
            ax.set_xlabel(r"$\Delta_p$ / kHz")
        ax.set_ylabel(r"Probe Transmission")
        ax.legend()
        plt.show()
        fig.canvas.mpl_connect('close_event', self.save_dialog)
        
    def p_plotter(self, dlist, plist):
        """
        Creates the figure and legend for a given array of steady state
        populations and probe detunings
        """
        if self.system_choice.currentIndex() == 0:
            self.spontaneous_21 = self.spontaneous_21_413
            self.spontaneous_32 = self.spontaneous_32_413
        if self.system_choice.currentIndex() == 1:
            self.spontaneous_21 = self.spontaneous_21_318
            self.spontaneous_32 = self.spontaneous_32_318
        self.dlist = dlist
        self.flist = plist
        self.typ = "p"
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title(f"{self.state_number} population")
        if dlist[-1] - dlist[0] >= 1e6:
            ax.plot(dlist/(1e6), plist, color="purple", label="$\Omega_c=$" f"{self.vals['omega_c']:.2e} $Hz$"\
                "\n" "$\Omega_p=$" f"{self.vals['omega_p']:.2e} $Hz$" "\n" \
                "$\Gamma_{c}$" f"= {self.spontaneous_32/(2*np.pi):.2e} $Hz$" "\n" \
                "$\Gamma_{p}$" f"= {self.spontaneous_21/(2*np.pi):.2e} $Hz$" "\n"\
                "$\Delta_c =$" f"{self.vals['delta_c']/1e6:.2f} $Hz$" "\n" \
                f"$\gamma_p$ = {self.vals['lwp']:.2e} $Hz$" "\n" 
                f"$\gamma_c$ = {self.vals['lwc']:.2e} $Hz$")
            ax.set_xlabel(r"$\Delta_p$ / MHz")
        else:
            ax.plot(dlist/(1e3), plist, color="purple", label="$\Omega_c=$" f"{self.vals['omega_c']:.2e} $Hz$"\
                "\n" "$\Omega_p=$" f"{self.vals['omega_p']:.2e} $Hz$" "\n" \
                "$\Gamma_{c}$" f"= {self.spontaneous_32/(2*np.pi):.2e} $Hz$" "\n" \
                "$\Gamma_{p}$" f"= {self.spontaneous_21/(2*np.pi):.2e} $Hz$" "\n"\
                "$\Delta_c =$" f"{self.vals['delta_c']/1e6:.2f} $Hz$" "\n" \
                f"$\gamma_p$ = {self.vals['lwp']:.2e} $Hz$" "\n" 
                f"$\gamma_c$ = {self.vals['lwc']:.2e} $Hz$")
            ax.set_xlabel(r"$\Delta_p$ / kHz")
        ax.set_ylabel(f"{self.state_number} state popultaion probability")
        ax.legend()
        plt.show()
        fig.canvas.mpl_connect('close_event', self.save_dialog)

    def n_plotter(self, dlist, nlist):
        """
        Creates the figure and legend for a given array of probe 
        transmission values and probe detunings
        """
        if self.system_choice.currentIndex() == 0:
            self.spontaneous_21 = self.spontaneous_21_413
            self.spontaneous_32 = self.spontaneous_32_413
        if self.system_choice.currentIndex() == 1:
            self.spontaneous_21 = self.spontaneous_21_318
            self.spontaneous_32 = self.spontaneous_32_318
        self.dlist = dlist
        self.flist = nlist
        print(FWHM(dlist, nlist))
        self.typ = "n"
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        """ Geometric library to calculate linewidth of EIT peak (FWHM) """
        max_gi = np.max(nlist)
        group_vel = c/max_gi
        if max_gi != 0:
            ax.text(0.8, 0.07, f"Max $n_g$ = {max_gi:.0f}", transform=ax.transAxes, fontsize=10, va='center', ha='center')
            ax.text(0.8, 0.13, f"Min $v_g$ = {group_vel:.0f} $m/s$", transform=ax.transAxes, fontsize=10, va='center', ha='center')                     
        plt.title(r"Group refractive index against probe beam detuning")
        if dlist[-1]-dlist[0] >= 1e6:
            ax.plot(dlist/(1e6), nlist, color="orange", label="$\Omega_c=$" f"{self.vals['omega_c']:.2e} $Hz$"\
                "\n" "$\Omega_p=$" f"{self.vals['omega_p']:.2e} $Hz$" "\n" \
                "$\Gamma_{c}$" f"= {self.spontaneous_32/(2*np.pi):.2e} $Hz$" "\n" \
                "$\Gamma_{p}$" f"= {self.spontaneous_21/(2*np.pi):.2e} $Hz$" "\n"\
                "$\Delta_c =$" f"{self.vals['delta_c']/1e6:.2f} $Hz$" "\n" \
                f"$\gamma_p$ = {self.vals['lwp']:.2e} $Hz$" "\n" 
                f"$\gamma_c$ = {self.vals['lwc']:.2e} $Hz$")
            ax.set_xlabel(r"$\Delta_p$ / MHz")
        else:
            ax.plot(dlist/(1e3), nlist, color="orange", label="$\Omega_c=$" f"{self.vals['omega_c']:.2e} $Hz$"\
                "\n" "$\Omega_p=$" f"{self.vals['omega_p']:.2e} $Hz$" "\n" \
                "$\Gamma_{c}$" f"= {self.spontaneous_32/(2*np.pi):.2e} $Hz$" "\n" \
                "$\Gamma_{p}$" f"= {self.spontaneous_21/(2*np.pi):.2e} $Hz$" "\n"\
                "$\Delta_c =$" f"{self.vals['delta_c']/1e6:.2f} $Hz$" "\n" \
                f"$\gamma_p$ = {self.vals['lwp']:.2e} $Hz$" "\n" 
                f"$\gamma_c$ = {self.vals['lwc']:.2e} $Hz$")
            ax.set_xlabel(r"$\Delta_p$ / kHz")
        ax.set_ylabel(r"Group Index")
        ax.legend()
        plt.show()
        fig.canvas.mpl_connect('close_event', self.save_dialog)
        
    def dme(self):
        """
        Brings up a warning box if the dipole matrix element for
        an entered n level state is not found
        """
        QMessageBox.about(self, "Dipole Matrix Element", 
        "Dipole matrix element not found! \n \nConsult readme for valid n levels")

    def power_warn(self):
        """
        Brings up a warning box if laser powers enetered incorrectly
        """
        QMessageBox.about(self, "Powers and Diameters", 
        "Please enter valid laser powers and diameters")

    def intensity_warn(self):
        """
        Brings up a warning box if laser intensities enetered incorrectly
        """
        QMessageBox.about(self, "Intensities", 
        "Please enter valid laser intensities")

    def transit_warn(self):
        """
        Brings up a warning box if laser diameters enetered incorrectly
        and transit time is selected
        """
        QMessageBox.about(self, "Transit time", 
        "Transit time cannot be included without specifying laser diameters")
        
    def rabi_warn(self):
        """
        Brings up a warning box if Rabi frequencies enetered incorrectly
        """
        QMessageBox.about(self, "Rabi frequencies", 
        "Please enter valid Rabi frequencies")
        
def main():        
    app = QApplication(sys.argv)
    window = UI()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    set_start_method("spawn")
    main()
    