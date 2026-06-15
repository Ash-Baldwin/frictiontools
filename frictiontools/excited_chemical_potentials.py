from scipy.optimize import brentq
import numpy as np
from frictiontools.utils import fermi_pop, excited_fermi_pop
from frictiontools.constants import boltzmann_kB

def converge_chemical_potential(
    temperature,
    target_electron_deviation,
    evs,
    ground_state_mu,
    fermi_mode,
    band ="valance",
    energy_window=None,
    valance_energies=None,
    mu_bounds=None,
    tolerance=1e-8
):
    """
    Converge on a new chemical potential that produces a target electron deviation 
    from the ground state Fermi-Dirac occupation.
    
    Parameters:
    -----------
    temperature : float
        Temperature in Kelvin
    target_electron_deviation : float
        Target number of electrons to deviate from ground state (can be positive or negative)
    evs : np.ndarray
        Energy eigenvalues in the desired energy window
    ground_state_mu : float
        Ground state chemical potential
    energy_window : tuple, optional
        (E_min, E_max) to restrict eigenvalues. If None, uses full evs array.
    fermi_mode : str, default "diff"
        Fermi distribution mode ("diff", "derivative", "excited", etc.)
    valance_energies : tuple, optional
        Required for fermi_mode="excited". Contains (VBM, CBM).
    mu_bounds : tuple, optional
        (mu_min, mu_max) bounds for root finding. If None, uses adaptive bounds.
    tolerance : float, default 1e-8
        Convergence tolerance for root finding
    
    Returns:
    --------
    float
        Converged chemical potential that produces the target electron deviation
    
    Raises:
    -------
    ValueError
        If root cannot be bracketed or other convergence issues occur
    """

    if band == "valance":
        temp = temperature[0]
    elif band == "conduction":
        temp = temperature[1]
    else:
        raise ValueError("band can only be either 'valance' or 'conduction'")

    ### Filter eigenvalues to energy window if specified
    if energy_window is not None:
        E_min, E_max = energy_window
        filtered_evs = evs[(evs >= E_min) & (evs <= E_max)]
    else:
        filtered_evs = np.asarray(evs)
    
    if len(filtered_evs) == 0:
        raise ValueError("No eigenvalues found in the specified energy window")
    
    ### Calculate ground state population using correct chemical potential component
    ground_pop = np.sum([fermi_pop(e, ground_state_mu, temp) for e in filtered_evs])

    ### Define objective function: electron deviation at test chemical potential

    def electron_deviation_function(test_mu):
        """
        Calculate electron population deviation distance from target.
        When this equals zero, we have found the target deviation.
        """
        if fermi_mode == "excited":
            test_pop = np.sum([excited_fermi_pop(e, valance_energies, [test_mu if band == "valance" else ground_state_mu, test_mu if band == "conduction" else ground_state_mu], temperature) for e in filtered_evs])
        else:
            test_pop = np.sum([fermi_pop(e, test_mu, temp) for e in filtered_evs])
        
        # Return the difference from the target
        return (test_pop - ground_pop) - target_electron_deviation
    

    ### Set root-finding bounds if not provided
    if mu_bounds is None:
        energy_range = evs.max() - evs.min()
        mu_min = evs.min() - 2 * energy_range
        mu_max = evs.max() + 2 * energy_range
    else:
        mu_min, mu_max = mu_bounds
    
    ### Check that the function has opposite signs at the bounds
    f_min = electron_deviation_function(mu_min)
    f_max = electron_deviation_function(mu_max)
    
    if f_min * f_max > 0:
        raise ValueError(
            f"Cannot bracket the root: f(mu_min={mu_min}) = {f_min}, "
            f"f(mu_max={mu_max}) = {f_max}. Both have the same sign. "
            f"Consider widening mu_bounds or checking the target_electron_deviation."
        )
    

    ### Use Brent's method for robust root finding
    converged_mu = brentq(
        electron_deviation_function,
        mu_min,
        mu_max,
        xtol=tolerance
    )
    
    
    return converged_mu