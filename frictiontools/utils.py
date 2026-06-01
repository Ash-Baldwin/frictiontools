import os
import numpy as np
from frictiontools.constants import boltzmann_kB

def gaussian_function(x,x0,s):
    return 1./np.sqrt(2*np.pi*s*s)*np.exp(-0.5*(((x-x0) / (s))**2))

def fermi_pop(x, x0, T):
    """
    Compute the Fermi population based on the given parameters.

    Parameters:
        x (float): Energy value.
        x0 (float): Fermi level.
        T (float): Temperature.
        n_spin (int): Number of spins.

    Returns:
        float: Fermi population.
    """

    if T < np.finfo(float).tiny:  # Check if temperature is near zero
        if x < x0:
            return 1.0  # Fermi population for T = 0
        else:
            return 0.0
    else:
        fermi_exponent = (x - x0) / (boltzmann_kB * T)
        if fermi_exponent > 100.0:  # Prevent floating-point overflow for large exponents
            return 0.0
        else:
            fermi_denominator = np.exp(fermi_exponent) + 1
            return 1 / fermi_denominator

def excited_fermi_pop(energy, valance_energies, chem_pot, temperature):
    """
    Compute the population due to an excited state Fermi distribution based on the given parameters.

    Parameters:
        e (float): Energy value.
        valance_energies (tuple): tuple containing VBM and CBM in eV (k-dependent).
        chem_pot (tuple): tuple containing Valance and Conduction band chemical potentials in eV (k-dependent).
        temperature (tuple): tuple containing Valance and Conduction band temperatures in K.

    Returns:
        float: Fermi population.
    """

    e_v = valance_energies[0]
    e_c = valance_energies[1]
    
    mu_v = chem_pot[0]
    mu_c = chem_pot[1]

    T_v = temperature[0]
    T_c = temperature[1]

    if energy <= e_v:
        fermi_exponent = (energy - mu_v) / (boltzmann_kB * T_v)

        if fermi_exponent > 100.0:  # Prevent floating-point overflow for large exponents
            return 0.0
        else:
            fermi_denominator = np.exp(fermi_exponent) + 1
            return 1 / fermi_denominator
    
    elif energy >= e_c:
        fermi_exponent = (energy - mu_c) / (boltzmann_kB * T_c)

        if fermi_exponent > 100.0:  # Prevent floating-point overflow for large exponents
            return 0.0
        else:
            fermi_denominator = np.exp(fermi_exponent) + 1
            return 1 / fermi_denominator
    
    else:
        return 0.0
    
        
def fermi_derivative(energy, chem_pot, temperature):
    """
    Calculate the derivative of the Fermi-Dirac distribution with respect to energy.

    Args:
        energy (float or ndarray): Energy value(s) in eV.
        chem_pot (float): Chemical potential (Fermi energy) in eV.
        temperature (float): Temperature in Kelvin.

    Returns:
        float or ndarray: The derivative of the Fermi-Dirac distribution.
    """
    # Boltzmann constant in eV/K
    boltzmann_kB = 8.617333262145e-5  # Boltzmann constant in eV/K

    # Fermi-Dirac distribution
    f = 1.0 / (np.exp((energy - chem_pot) / (boltzmann_kB * temperature)) + 1.0)

    # Derivative of Fermi-Dirac distribution
    f_derivative = -f * (1 - f) / (boltzmann_kB * temperature)
    
    return f_derivative

def excited_fermi_derivative(e, e_v, e_c, mu_v, mu_c, T_v, T_c):
    """
    Calculate the derivative of the Fermi-Dirac distribution with respect to energy.

    Args:
        energy (float or ndarray): Energy value(s) in eV.
        chem_pot (float): Chemical potential (Fermi energy) in eV.
        temperature (float): Temperature in Kelvin.

    Returns:
        float or ndarray: The derivative of the Fermi-Dirac distribution.
    """
    # Boltzmann constant in eV/K
    boltzmann_kB = 8.617333262145e-5  # Boltzmann constant in eV/K

    if e <= e_v:
        # Fermi-Dirac distribution
        f = 1.0 / (np.exp((e - mu_v) / (boltzmann_kB * T_v)) + 1.0)

        # Derivative of Fermi-Dirac distribution
        f_derivative = -f * (1 - f) / (boltzmann_kB * T_v)
    elif e >= e_c:
        # Fermi-Dirac distribution
        f = 1.0 / (np.exp((e - mu_c) / (boltzmann_kB * T_c)) + 1.0)

        # Derivative of Fermi-Dirac distribution
        f_derivative = -f * (1 - f) / (boltzmann_kB * T_c)
    else:
        f_derivative = 0.0
    
    return f_derivative

def find_fermi_occupation(evs, chem_pot, temp, min_state, max_state):
    occ = 0.0
    for i in range(min_state, max_state):
        occ += fermi_pop(evs[i], chem_pot, temp)
    
    return occ

def find_excited_fermi_occupation(evs, chem_pot, valance_energies, temp, min_state, max_state):
    occ = 0.0
    for i in range(min_state, max_state):
        occ += excited_fermi_pop(evs[i], valance_energies, chem_pot, temp)
    
    return occ