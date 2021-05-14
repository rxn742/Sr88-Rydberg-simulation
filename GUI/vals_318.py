#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 23:15:45 2021

@author: robgc
"""
import numpy as np
from scipy.constants import hbar, e, epsilon_0,c, m_e
from sys import path
path.insert(0, '../utility')
from lifetime_fits import state_3S1, state_3D3
import csv

""" 413nm Strontium """
d23_318 = np.sqrt((3*2.9e-6*hbar*318e-9*e**2)/(4*np.pi*m_e*c)) # r-i dipole matrix element
d12_318 = 1.339579873e-30 # i-g dipole matrix element
spontaneous_32_318 = 5.1e4 # Spontaneous emission rate from r to i
spontaneous_21_318 = 4.69e4 # Spontaneous emission rate from i to g
kp_318 = 2*np.pi/689e-9 # Probe wavevector in m^-1
kc_318 = 2*np.pi/318e-9 # Coupling wavevector in m^-1

os_3D3 = {}
wl_3D3 = {}
with open("rydberg_vals.csv", "rt") as file: 
    reader = csv.reader(file, delimiter=',')
    for rows in reader:
        try:
            os_3D3[str(rows[0])] = float(rows[2])
            wl_3D3[str(rows[0])] = float(rows[4])
        except:
            pass

def func_omega_c_318(Ic, d_23):
    """
    Calculates coupling Rabi frequency for a given intensity
    and transition dipole matrix element
    """
    return (d_23/hbar)*np.sqrt((2*Ic)/(c*epsilon_0))

def func_omega_p_318(Ip):
    """
    Calculates probe Rabi frequency for a given intensity
    """
    return (d12_318/hbar)*np.sqrt((2*Ip)/(c*epsilon_0))

def func_Ic_318(cp, cd):
    """
    Calculates intensity for a given laser power and diameter
    """
    return cp/(np.pi*(cd/2)**2)

def func_Ip_318(pp, pd):
    """
    Calculates intensity for a given laser power and diameter
    """
    return pp/(np.pi*(pd/2)**2)

def func_spon_318(n, series):
    """
    Returns the spontaneous emission rate of a given Rydberg state
    """
    if series == "3S1":
        return 1/state_3S1(n)
    if series == "3D3":
        return 1/state_3D3(n)
    
def func_d23_318(n, series):
    """
    Returns the transition dipole matrix element of a given Rydberg state
    """
    if series == "3D3":
        try:    
            return np.sqrt((3*os_3D3[str(n)]*wl_3D3[str(n)]*1e-9*hbar*e**2)/(4*np.pi*m_e*c))
        except:
            return 0

def func_kc_318(n, series):
    """
    Returns the wavenumber of a given Rydberg state
    """
    if series == "3D3":
        try:
            return 2*np.pi/(wl_3D3[str(n)]*1e-9)
        except:
            return 0
