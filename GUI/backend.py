#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:51:34 2020

@author: robgc
"""
from miniutils import parallel_progbar
import qutip as qt
qt.settings.auto_tidyup=False
import numpy as np
from scipy.constants import hbar, epsilon_0, k, c
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.signal import find_peaks, peak_widths, peak_prominences

"""defining the states"""
state1 = qt.basis(3,0) # ground state
state2 = qt.basis(3,1) # intermediate state
state3 = qt.basis(3,2) # excited state

"""defining identities"""
state1_operator = state1*state1.dag()
state2_operator = state2*state2.dag()
state3_operator = state3*state3.dag()

"""defining transition operators"""
transition_12 = state2*state1.dag() # ground to intermediate
transition_21 = state1*state2.dag() # intermediate to ground
transition_23 = state3*state2.dag() # intermediate to excited
transition_32 = state2*state3.dag() # excited to intermediate

"""defining functions"""
def hamiltonian(delta_p, delta_c, omega_p, omega_c):
    """
    This function defines the Hamiltonian of the 3 level system
    Parameters
    ----------
    delta_p : float
        Probe detuning in rad/s
    delta_c : float
        Coupling detuning in rad/s
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s

    Returns
    -------
    Qutip.Qobj (operator)
        Hamiltionian of the system (Uses Qutip convention hbar = 1)

    """
    return (-delta_p*(state2_operator + state3_operator) \
            - delta_c*(state3_operator) \
            + omega_p*(transition_21 + transition_12)/2 \
            + omega_c*(transition_32 + transition_23)/2)    

def spontaneous_emission(spontaneous_32, spontaneous_21):
    """
    This function defines the spntaneous emission collapse operators
    Parameters
    ----------
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    
    Returns
    -------
    list, dtype = float
        List of spontaneous emission rates

    """

    return [np.sqrt(spontaneous_32)*transition_32, \
            np.sqrt(spontaneous_21)*transition_21]
            
def laser_linewidth(lw_probe, lw_coupling):
    """
    Parameters
    ----------
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s

    Returns
    -------
    lw : numpy.ndarray, shape = 9x9, dtype = float64
        The laser linewidth super operator 

    """
    lw = np.zeros((9,9))
    lw[1,5] = -lw_probe
    lw[2,2] = -lw_probe-lw_coupling
    lw[3,3] = -lw_probe
    lw[5,5] = -lw_coupling
    lw[6,6] = -lw_probe-lw_coupling
    lw[7,7] = -lw_coupling
    return lw

def transit_time(temperature, probe_diameter, coupling_diameter):
    """
    Parameters
    ----------
    temperature : float
        Temperature of the oven in Kelvin
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres

    Returns
    -------
    tt : numpy.ndarray, shape = 9x9, dtype = float64
        The transit time super operator 

    """
    mean_speed = 0.75*np.sqrt(np.pi)*u(temperature)
    tt_probe = mean_speed/probe_diameter
    tt_coupling = mean_speed/coupling_diameter
    tt_array = np.zeros((9,9))
    if temperature == 0:
        return tt_array
    else:
        tt_array[1,5] = -tt_probe
        tt_array[2,2] = -tt_probe-tt_coupling
        tt_array[3,3] = -tt_probe
        tt_array[5,5] = -tt_coupling
        tt_array[6,6] = -tt_probe-tt_coupling
        tt_array[7,7] = -tt_coupling
        return tt_array    

def Liouvillian(delta_p, delta_c, omega_p, omega_c,
                spontaneous_32, spontaneous_21, lw_probe, lw_coupling, 
                temperature, probe_diameter, coupling_diameter, tt):
    """
    This function calculates the Liouvillian of the system 
    Parameters
    ----------
    delta_p : float
        Probe detuning in rad/s
    delta_c : float
        Coupling detuning in rad/s
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s
    temperature : float
        Temperature of the oven in Kelvin
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres
    tt : string
        Enter argument "Y" for transit time to be included

    Returns
    -------
    L : Qutip.Qobj (super)
        The full Liouvillian super operator of the system for the master eqn

    """
    H = hamiltonian(delta_p, delta_c, omega_p, omega_c)
    c_ops = spontaneous_emission(spontaneous_32, spontaneous_21)
    L = qt.liouvillian(H, c_ops)
    L_arr = L.data.toarray() # change to numpy array to add on laser linewidth matrix
    L_arr += laser_linewidth(lw_probe, lw_coupling)
    if tt == "Y":
        L_arr += transit_time(temperature, probe_diameter, coupling_diameter)
    L = qt.Qobj(L_arr, dims=[[[3], [3]], [[3], [3]]], type="super") # change back to Qobj
    return L

def population(delta_p, delta_c, omega_p, omega_c, spontaneous_32, spontaneous_21,
               lw_probe, lw_coupling, temperature, probe_diameter, coupling_diameter, tt):
    """
    This function solves for the steady state density matrix of the system
    Parameters
    ----------
    delta_p : float
        Probe detuning in rad/s
    delta_c : float
        Coupling detuning in rad/s
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s
    temperature : float
        Temperature of the oven in Kelvin
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres
    tt : string
        Enter argument "Y" for transit time to be included

    Returns
    -------
    rho : Qutip.Qobj (Density Matrix)
        The steady state density matrix of the 3 level system

    """
    rho = qt.steadystate(Liouvillian(delta_p, delta_c, omega_p, omega_c, 
                                     spontaneous_32, spontaneous_21, lw_probe, 
                                     lw_coupling, temperature, probe_diameter, coupling_diameter, tt))
    return rho

def doppler(v, delta_p, delta_c, omega_p, omega_c, spontaneous_32,
            spontaneous_21, lw_probe, lw_coupling, sigma, kp, kc, state_index, 
            temperature, probe_diameter, coupling_diameter, tt):
    """
    This function generates the integrand to solve when including Doppler broadening
    Parameters
    ----------
    v : float
        Transverse velocity of atom in m/s
    delta_p : float
        Probe detuning in rad/s
    delta_c : float
        Coupling detuning in rad/s
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s
    sigma : float
        Width of the transverse velocity distribution in m/s
    kp : float
        Probe transition wavenumber in m^-1
    kc : float
        Coupling transition wavenumber in m^-1
    state_index : tuple
        Element of density matrix to select
    temperature : float
        Temperature of the oven in Kelvin
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres
    tt : string
        Enter argument "Y" for transit time to be included
        
    Returns
    -------
    integrand : float
        Gaussian weighted integrand

    """
    if state_index == (1,0):
        integrand = np.imag(population(delta_p-kp*v, delta_c+kc*v, omega_p, 
        omega_c, spontaneous_32, spontaneous_21, lw_probe, lw_coupling, 
        temperature, probe_diameter, coupling_diameter, tt)[state_index]*maxwell_trans(v, sigma))
    else:
        integrand = np.real(population(delta_p-kp*v, delta_c+kc*v, omega_p, 
        omega_c, spontaneous_32, spontaneous_21, lw_probe, lw_coupling, 
        temperature, probe_diameter, coupling_diameter, tt)[state_index]*maxwell_trans(v, sigma))
    return integrand

def doppler_int(delta_p, delta_c, omega_p, omega_c, spontaneous_32, 
               spontaneous_21, lw_probe, lw_coupling, sigma, kp, kc, state_index, beamdiv, 
               temperature, probe_diameter, coupling_diameter, tt):
    """
    This function generates the integrand to solve when including Doppler broadening
    Parameters
    ----------
    delta_p : float
        Probe detuning in rad/s
    delta_c : float
        Coupling detuning in rad/s
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s
    sigma : float
        Width of the transverse velocity distribution in m/s
    kp : float
        Probe transition wavenumber in m^-1
    kc : float
        Coupling transition wavenumber in m^-1
    state_index : tuple
        Element of density matrix to select
    temperature : float
        Temperature of the oven in Kelvin
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres
    tt : string
        Enter argument "Y" for transit time to be included
        
    Returns
    -------
    p_avg : float
        Doppler averaged density matrix element

    """
    p_avg = quad(doppler, -3*sigma, 3*sigma, args=(delta_p, delta_c, omega_p, omega_c, spontaneous_32,
            spontaneous_21, lw_probe, lw_coupling, sigma, kp, kc, state_index, 
            temperature, probe_diameter, coupling_diameter, tt))[0]
    return p_avg
    
def pop_calc(delta_c, omega_p, omega_c, spontaneous_32, 
             spontaneous_21, lw_probe, lw_coupling, dmin, dmax, steps, 
             state_index, gauss, temperature, kp, kc, beamdiv, 
             probe_diameter, coupling_diameter, tt):
    """
    This function generates an array of population values 
    for a generated list of probe detunings
    Parameters
    ---------- 
    delta_c : float
        Coupling detuning in Hz.
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s
    dmin : float
        Lower bound of Probe detuning in angular MHz
    dmax : float
        Upper bound of Probe detuning in angular MHz
    steps : int
        Number of Probe detunings to calculate the population probability
    state_index : tuple
        Element of density matrix to select
    gauss : string
        Enter argument "Y" to include Doppler broadening
    temperature : float
        Temperature of the oven in Kelvin
    kp : float
        Probe transition wavenumber in m^-1
    kc : float
        Coupling transition wavenumber in m^-1
    beamdiv : float
        Full divergence angle of the atomic beam in rad
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres
    tt : string
        Enter argument "Y" for transit time to be included

    Returns
    -------
    dlist : numpy.ndarray, dtype = float64
        Array of Probe detunings
    plist : numpy.ndarray, dtype = float64
        Array of population probabilities corresponding to the detunings

    """
    iters = np.empty(steps+1, dtype=tuple)
    dlist = np.linspace(dmin, dmax, steps+1)
    if gauss == "Y":
        sigma = beamdiv*np.sqrt(3/2)*u(temperature)
        for i in range(0, steps+1):
            iters[i] = (dlist[i], delta_c, omega_p, omega_c, 
                        spontaneous_32, spontaneous_21, 
                        lw_probe, lw_coupling, sigma, 
                        kp, kc, state_index, beamdiv, 
                        temperature, probe_diameter, 
                        coupling_diameter, tt)
        plist = np.abs(np.array(parallel_progbar(doppler_int, iters, starmap=True)))
    else:
        for i in range(0, steps+1):
            iters[i] = (dlist[i], delta_c, omega_p, omega_c, 
                        spontaneous_32, spontaneous_21, lw_probe, 
                        lw_coupling, temperature, probe_diameter, 
                        coupling_diameter, tt)
        plist = parallel_progbar(population, iters, starmap=True)
        plist = np.array(np.abs([x[state_index] for x in plist]))
    return dlist, plist

def refractiveindex(delta_p, delta_c, omega_p, omega_c, spontaneous_32, 
                 spontaneous_21, lw_probe, lw_coupling, density, dig, kp, sl, 
                 temperature, probe_diameter, coupling_diameter, tt):
    """
    This function calculates a refractive index for a given set of parameters
    Parameters
    ----------
    delta_p : float
        Probe detuning in rad/s
    delta_c : float
        Coupling detuning in rad/s
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s
    density : float
        Number density of atoms in the sample.
    dig : float
        Probe transition diple matrix element in Cm
    kp : float
        Probe transition wavenumber in m^-1
    sl : float
        Atomic beam diameter
    temperature : float
        Temperature of the oven in Kelvin
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres
    tt : string
        Enter argument "Y" for transit time to be included    
    Returns
    -------
    T : float
        Relative probe transmission value for the given parameters

    """
    p = population(delta_p, delta_c, omega_p, omega_c, spontaneous_32, 
                   spontaneous_21, lw_probe, lw_coupling, temperature, 
                   probe_diameter, coupling_diameter, tt)[1,0] # element rho_ig
    chi = (-2*density*dig**2*p)/(hbar*epsilon_0*omega_p) # calculate susceptibility
    a = kp*np.abs(chi.imag) # absorption coefficient
    
    n_real = np.sqrt(1+np.real(chi)) #phase index of refraction
    

    return n_real


def ncalc(delta_c, omega_p, omega_c, spontaneous_32, 
             spontaneous_21, lw_probe, lw_coupling, dmin, dmax, steps, 
             gauss, kp, kc, density, dig, sl, temperature, beamdiv, probe_diameter, coupling_diameter, tt):
    """
    This function generates an array of group refractive indicies for a generated list of probe detunings
    Parameters
    ---------- 
    delta_c : float
        Coupling detuning in rad/s
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s
    dmin : float
        Lower bound of Probe detuning in angular MHz
    dmax : float
        Upper bound of Probe detuning in angular MHz
    steps : int
        Number of Probe detunings to calculate the population probability
    gauss : string
        Enter argument "Y" to include Doppler broadening
    kp : float
        Probe transition wavenumber in m^-1
    kc : float
        Coupling transition wavenumber in m^-1
    density : float
        Number density of atoms in the sample.
    dig : float
        Probe transition diple matrix element in Cm
    sl : float
        Atomic beam diameter
    temperature : float
        Temperature of the oven in Kelvin
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres
    tt : string
        Enter argument "Y" for transit time to be included   

    Returns
    -------
    dlist : numpy.ndarray, dtype = float64
        Array of Probe detunings
    tlist : numpy.ndarray, dtype = float64
        Array of transmission values corresponding to the detunings

    """

    iters = np.empty(steps+1, dtype=tuple)
    dlist = np.linspace(dmin, dmax, steps+1)
    
# =============================================================================
#     if gauss == "Y":
#         sigma = beamdiv*np.sqrt(3/2)*u(temperature)
#         elem = 1,0
#         for i in range(0, steps+1):
#             iters[i] = (dlist[i], delta_c, omega_p, omega_c, 
#                         spontaneous_32, spontaneous_21, lw_probe, 
#                         lw_coupling, sigma, kp, kc, elem, beamdiv, 
#                         temperature, probe_diameter, coupling_diameter, tt)
#         rhos = np.array(parallel_progbar(doppler_int, iters, starmap=True))
#         chi_imag = (-2*density*dig**2*rhos)/(hbar*epsilon_0*omega_p)
#         a = kp*np.abs(chi_imag)
#         tlist = np.exp(-a*sl)
#     else:
# =============================================================================
    for i in range(0, steps+1):
        iters[i] = (dlist[i], delta_c, omega_p, omega_c, 
                    spontaneous_32, spontaneous_21, lw_probe, 
                    lw_coupling, density, dig, kp, sl, temperature, 
                    probe_diameter, coupling_diameter, tt)
    n_real = np.array(parallel_progbar(transmission, iters, starmap=True))
    
    w_21 = c * 2*np.pi/(461e-9) # probe transition frequency
    
    n_g = n_real[:-1] + (dlist[:-1] + w_21) * np.diff(n_real)/np.diff(dlist)
    return dlist[:-1], n_g

def transmission(delta_p, delta_c, omega_p, omega_c, spontaneous_32, 
                 spontaneous_21, lw_probe, lw_coupling, density, dig, kp, sl, 
                 temperature, probe_diameter, coupling_diameter, tt):
    """
    This function calculates a transmission value for a given set of parameters
    Parameters
    ----------
    delta_p : float
        Probe detuning in rad/s
    delta_c : float
        Coupling detuning in rad/s
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s
    density : float
        Number density of atoms in the sample.
    dig : float
        Probe transition diple matrix element in Cm
    kp : float
        Probe transition wavenumber in m^-1
    sl : float
        Atomic beam diameter
    temperature : float
        Temperature of the oven in Kelvin
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres
    tt : string
        Enter argument "Y" for transit time to be included    
    Returns
    -------
    T : float
        Relative probe transmission value for the given parameters

    """
    p = population(delta_p, delta_c, omega_p, omega_c, spontaneous_32, 
                   spontaneous_21, lw_probe, lw_coupling, temperature, 
                   probe_diameter, coupling_diameter, tt)[1,0] # element rho_ig
    chi = (-2*density*dig**2*p)/(hbar*epsilon_0*omega_p) # calculate susceptibility
    a = kp*np.abs(chi.imag) # absorption coefficient
    T = np.exp(-a*sl)
    return T


def tcalc(delta_c, omega_p, omega_c, spontaneous_32, 
             spontaneous_21, lw_probe, lw_coupling, dmin, dmax, steps, 
             gauss, kp, kc, density, dig, sl, temperature, beamdiv, probe_diameter, coupling_diameter, tt):
    """
    This function generates an array of transmission values for a generated list of probe detunings
    Parameters
    ---------- 
    delta_c : float
        Coupling detuning in rad/s
    omega_p : float
        Probe Rabi frequency in rad/s
    omega_c : float
        Coupling Rabi frequency in rad/s
    spontaneous_32 : float
        state 3 to state 2 spontaneous emission rate in rad/s
    spontaneous_21 : float
        state 2 to state 1 spontaneous emission rate in rad/s
    lw_probe : float
        Probe beam linewidth in rad/s
    lw_coupling : float
        Coupling beam linewidth rad/s
    dmin : float
        Lower bound of Probe detuning in angular MHz
    dmax : float
        Upper bound of Probe detuning in angular MHz
    steps : int
        Number of Probe detunings to calculate the population probability
    gauss : string
        Enter argument "Y" to include Doppler broadening
    kp : float
        Probe transition wavenumber in m^-1
    kc : float
        Coupling transition wavenumber in m^-1
    density : float
        Number density of atoms in the sample.
    dig : float
        Probe transition diple matrix element in Cm
    sl : float
        Atomic beam diameter
    temperature : float
        Temperature of the oven in Kelvin
    probe_diameter: float
        Circular probe laser diameter in metres
    coupling_diameter: float
        Circular coupling laser diameter in metres
    tt : string
        Enter argument "Y" for transit time to be included   

    Returns
    -------
    dlist : numpy.ndarray, dtype = float64
        Array of Probe detunings
    tlist : numpy.ndarray, dtype = float64
        Array of transmission values corresponding to the detunings

    """
    iters = np.empty(steps+1, dtype=tuple)
    dlist = np.linspace(dmin, dmax, steps+1)
    if gauss == "Y":
        sigma = beamdiv*np.sqrt(3/2)*u(temperature)
        elem = 1,0
        for i in range(0, steps+1):
            iters[i] = (dlist[i], delta_c, omega_p, omega_c, 
                        spontaneous_32, spontaneous_21, lw_probe, 
                        lw_coupling, sigma, kp, kc, elem, beamdiv, 
                        temperature, probe_diameter, coupling_diameter, tt)
        rhos = np.array(parallel_progbar(doppler_int, iters, starmap=True))
        chi_imag = (-2*density*dig**2*rhos)/(hbar*epsilon_0*omega_p)
        a = kp*np.abs(chi_imag)
        tlist = np.exp(-a*sl)
    else:
        for i in range(0, steps+1):
            iters[i] = (dlist[i], delta_c, omega_p, omega_c, 
                        spontaneous_32, spontaneous_21, lw_probe, 
                        lw_coupling, density, dig, kp, sl, temperature, 
                        probe_diameter, coupling_diameter, tt)
        tlist = np.array(parallel_progbar(transmission, iters, starmap=True))
    return dlist, tlist
    
def FWHM(dlist, tlist):
    """
    This function calculates the FWHM of the EIT peak in a spectrum
    Parameters
    ----------
    dlist : numpy.ndarray, dtype = float64
        Array of Probe detunings
    tlist : numpy.ndarray, dtype = float64
        Array of transmission values corresponding to the detunings

    Returns
    -------
    Peak width : float
        The FWHM of the EIT Peak in angular MHz

    """
    peak = find_peaks(tlist, distance = 999)[0]
    sample = dlist[1]-dlist[0]
    width = peak_widths(tlist, peak)[0]*sample
    try:
        return width[0]
    except:
        return 0

def contrast(dlist, tlist):
    """
    This function calculates the contrast of the EIT peak in a spectrum
    Parameters
    ----------
    dlist : numpy.ndarray, dtype = float64
        Array of Probe detunings
    tlist : numpy.ndarray, dtype = float64
        Array of transmission values corresponding to the detunings

    Returns
    -------
    Contrast : float
        The contrast of the EIT Peak

    """
    peak = find_peaks(tlist, distance = 999)[0]
    contrast = peak_prominences(tlist, peak)
    try:
        return contrast[0][0]
    except:
        return 0

def u(T):
    """
    This function calculates most probable speed of a 
    strontium atom moving in a gas
    ----------
    T : float
        Temperature of the oven in Kelvin

    Returns
    -------
    u: float
        Most probable speed of a strontium atom moving in a gas in m/s

    """
    return np.sqrt(2*k*(T)/(88*1.6605390666e-27))

def maxwell_long(v, u):
    """
    This function calculates the longitudinal velocity 
    distribution of strontium atoms moving in a beam
    ----------
    v : float
        Velocity of strontium atom in m/s
    u: float
        Most probable speed of a strontium atom moving in a gas in m/s

    Returns
    -------
    f(vl) : float
        Probability density of finding an atom with a given longitudinal velocity

    """
    return 2*(v**3/u**4)*np.exp(-(v**2/u**2))

def maxwell_trans(v, sigma):
    """
    This function calculates the transvers velocity 
    distribution of strontium atoms moving in a beam
    ----------
    v : float
        Velocity of strontium atom in m/s
    sigma: float
        Width of the transverse velocity distribution in m/s

    Returns
    -------
    f(vt) : float
        Probability density of finding an atom with a given transverse velocity

    """
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(v**2/(2*sigma**2)))

