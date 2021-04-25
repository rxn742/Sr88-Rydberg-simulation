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
from scipy.constants import hbar, epsilon_0, k
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
        Probe detuning in Hz.
    delta_c : float
        Coupling detuning in Hz.
    omega_p : float
        Probe Rabi frequency in Hz.
    omega_c : float
        Coupling Rabi frequency in Hz.

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
    gamma_32 : float
        state 3 to state 2 spontaneous emission rate.
    gamma_21 : float
        state 2 to state 1 spontaneous emission rate.
    
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
    lwp : float
        Probe beam linewidth in Hz
    lwc : float
        Coupling beam linewidth in Hz

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
    Temperature : float (Kelvin)
        Temperature of the oven

    Returns
    -------
    tt : numpy.ndarray, shape = 9x9, dtype = float64
        The transit time super operator 

    """
    mean_speed = 0.75*np.sqrt(np.pi)*v_mp(temperature)
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
        Probe detuning in Hz.
    delta_c : float
        Coupling detuning in Hz.

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
        Probe detuning in Hz.
    delta_c : float
        Coupling detuning in Hz.

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
            spontaneous_21, lw_probe, lw_coupling, mp, kp, kc, state_index, 
            temperature, probe_diameter, coupling_diameter, tt):
    """
    This function generates the integrand to solve when including Doppler broadening
    Parameters
    ----------
    v : float
        Transverse velocity of atom
    delta_p : float
        Probe detuning in Hz.
    delta_c : float
        Coupling detuning in Hz.
    mu : float
        Mean transverse velocity
    sig : float
        Transverse velocity standard deviation
    state_index : tuple
        chosen element of the density matrix
        
    Returns
    -------
    integrand : float
        Gaussian weighted integrand

    """
    if state_index == (1,0):
        integrand = np.imag(population(delta_p-kp*v, delta_c+kc*v, omega_p, 
        omega_c, spontaneous_32, spontaneous_21, lw_probe, lw_coupling, 
        temperature, probe_diameter, coupling_diameter, tt)[state_index]*maxwell_trans(v, mp))
    else:
        integrand = np.real(population(delta_p-kp*v, delta_c+kc*v, omega_p, 
        omega_c, spontaneous_32, spontaneous_21, lw_probe, lw_coupling, 
        temperature, probe_diameter, coupling_diameter, tt)[state_index]*maxwell_trans(v, mp))
    return integrand

def doppler_int(delta_p, delta_c, omega_p, omega_c, spontaneous_32, 
               spontaneous_21, lw_probe, lw_coupling, mp, kp, kc, state_index, beamdiv, 
               temperature, probe_diameter, coupling_diameter, tt):
    """
    This function generates the integrand to solve when including Doppler broadening
    Parameters
    ----------
    delta_p : float
        Probe detuning in Hz.
    delta_c : float
        Coupling detuning in Hz.
    mu : float
        Mean transverse velocity
    sig : float
        Transverse velocity standard deviation
    state_index : tuple
        chosen element of the density matrix
        
    Returns
    -------
    p_avg : float
        Doppler averaged density matrix element

    """
    mp = mp*beamdiv
    p_avg = quad(doppler, -3*mp, 3*mp, args=(delta_p, delta_c, omega_p, omega_c, spontaneous_32,
            spontaneous_21, lw_probe, lw_coupling, mp, kp, kc, state_index, 
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
    dmin : float
        Lower bound of Probe detuning in MHz
    dmax : float
        Upper bound of Probe detuning in MHz
    steps : int
        Number of Probe detunings to calculate the population probability
    state_index : tuple
        chosen element of the density matrix

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
        mp = np.sqrt(3/2)*v_mp(temperature)
        for i in range(0, steps+1):
            iters[i] = (dlist[i], delta_c, omega_p, omega_c, 
                        spontaneous_32, spontaneous_21, 
                        lw_probe, lw_coupling, mp, 
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

def pop_plot(state, delta_c, omega_p, omega_c, spontaneous_32, 
             spontaneous_21, lw_probe, lw_coupling, dmin, dmax, 
             steps, gauss, temperature, kp, kc, beamdiv, 
             probe_diameter, coupling_diameter, tt):
    """
    This function plots the population probability of a chosen state against probe detuning
    Parameters
    ---------- 
    delta_c : float
        Coupling detuning in Hz.
    dmin : float
        Lower bound of Probe detuning in MHz
    dmax : float
        Upper bound of Probe detuning in MHz
    steps : int
        Number of Probe detunings to calculate the population probabilities 

    Returns
    -------
    Population : plot
        Plot of chosen state population probability against probe detuning

    """
    if state == "Ground":
        state_index = 0,0
    if state == "Intermediate":
        state_index = 1,1
    if state == "Rydberg":
        state_index = 2,2
    
    dlist, plist = pop_calc(delta_c, omega_p, omega_c, spontaneous_32, 
                            spontaneous_21, lw_probe, lw_coupling, dmin, dmax, steps, state_index, 
                            gauss, temperature, kp, kc, beamdiv, probe_diameter, coupling_diameter, tt)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(f"{state} population")
    if dmax - dmin >= 1e6:
        ax.plot(dlist/(1e6), plist, color="orange", label="$\Omega_c=$" f"{omega_c:.2e} $Hz$"\
                "\n" "$\Omega_p=$" f"{omega_p:.2e} $Hz$" "\n" \
                "$\Gamma_{c}$" f"= {spontaneous_32/(2*np.pi):.2e} $Hz$" "\n" \
                "$\Gamma_{p}$" f"= {spontaneous_21/(2*np.pi):.2e} $Hz$" "\n"\
                "$\Delta_c =$" f"{delta_c/1e6:.2f} $Hz$" "\n" \
                f"$\gamma_p$ = {lw_probe:.2e} $Hz$" "\n" 
                f"$\gamma_c$ = {lw_coupling:.2e} $Hz$")
        ax.set_xlabel(r"$\Delta_p$ / MHz")
    else:
        ax.plot(dlist/(1e3), plist, color="orange", label="$\Omega_c=$" f"{omega_c:.2e} $Hz$"\
                "\n" "$\Omega_p=$" f"{omega_p:.2e} $Hz$" "\n" \
                "$\Gamma_{c}$" f"= {spontaneous_32/(2*np.pi):.2e} $Hz$" "\n" \
                "$\Gamma_{p}$" f"= {spontaneous_21/(2*np.pi):.2e} $Hz$" "\n"\
                "$\Delta_c =$" f"{delta_c/1e6:.2f} $Hz$" "\n" \
                f"$\gamma_p$ = {lw_probe:.2e} $Hz$" "\n" 
                f"$\gamma_c$ = {lw_coupling:.2e} $Hz$")
        ax.set_xlabel(r"$\Delta_p$ / kHz")
    ax.set_ylabel(f"{state} state popultaion")
    ax.legend()
    plt.show()

def transmission(delta_p, delta_c, omega_p, omega_c, spontaneous_32, 
                 spontaneous_21, lw_probe, lw_coupling, density, dig, kp, sl, 
                 temperature, probe_diameter, coupling_diameter, tt):
    """
    This function calculates a transmission value for a given set of parameters
    Parameters
    ----------
    density : float
        Number density of atoms in the sample.   
    delta_p : float
        Probe detuning in Hz.
    delta_c : float
        Coupling detuning in Hz.
    sl : float
        Atomic beam diameter

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
        Coupling detuning in Hz.
    dmin : float
        Lower bound of Probe detuning in MHz
    dmax : float
        Upper bound of Probe detuning in MHz
    steps : int
        Number of Probe detunings to calculate the transmission at 

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
        mp = np.sqrt(3/2)*v_mp(temperature)
        print(mp)
        elem = 1,0
        for i in range(0, steps+1):
            iters[i] = (dlist[i], delta_c, omega_p, omega_c, 
                        spontaneous_32, spontaneous_21, lw_probe, 
                        lw_coupling, mp, kp, kc, elem, beamdiv, 
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
    
def trans_plot(delta_c, omega_p, omega_c, spontaneous_32, 
             spontaneous_21, lw_probe, lw_coupling, dmin, dmax, steps, 
             gauss, kp, kc, density, dig, sl, temperature, beamdiv, 
             probe_diameter, coupling_diameter, tt):
    """
    This function plots probe beam transmission for an array of probe detunings
    Parameters
    ---------- 
    delta_c : float
        Coupling detuning in Hz.
    dmin : float
        Lower bound of Probe detuning in MHz
    dmax : float
        Upper bound of Probe detuning in MHz
    steps : int
        Number of Probe detunings to calculate the transmission at 

    Returns
    -------
    T : plot
        Plot of probe beam transmission against probe detuning, with EIT FWHM

    """
    dlist, tlist = tcalc(delta_c, omega_p, omega_c, spontaneous_32, 
                         spontaneous_21, lw_probe, lw_coupling, dmin, dmax, 
                         steps, gauss, kp, kc, density, dig, sl, temperature, beamdiv, 
                         probe_diameter, coupling_diameter, tt)

    
def FWHM(dlist, tlist):
    """
    This function calculates the FWHM of the EIT peak in a spectrum
    Parameters
    ----------
    t : numpy.ndarray, dtype = float
        Calculated transmission values for a range of detunings

    Returns
    -------
    pw : float
        The FWHM of the EIT Peak in MHz

    """
    peak = find_peaks(tlist, distance = 999)[0]
    sample = dlist[1]-dlist[0]
    width = peak_widths(tlist, peak)[0]*sample
    return width[0]

def contrast(dlist, tlist):
    peak = find_peaks(tlist, distance = 999)[0]
    contrast = peak_prominences(tlist, peak)
    return contrast[0][0]

def v_mp(T):
    return np.sqrt(2*k*(T)/(88*1.6605390666e-27))

def maxwell_long(v, mp):
    return 2*(v**3/mp**4)*np.exp(-(v**2/mp**2))

def maxwell_trans(v, mp):
    return 1/(np.sqrt(np.pi)*mp)*np.exp(-(v**2/mp**2))

def effectiveT(mp):
    return ((mp**2*(88*1.6605390666e-27))/(2*k))
