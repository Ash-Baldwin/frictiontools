import sys
sys.path.append("/fs/home/ash/Code/frictiontools")

import os
from os.path import join as pathjoin

import time
import numpy as np
import matplotlib.pyplot as plt

from frictiontools.aims_parse import get_friction_masses, get_friction_indices, extract_atomic_symbols, get_number_of_atoms, parse_kpoints_and_weights, parse_ev_data
from frictiontools.vib_parse import parse_normal_modes, parse_perturbing_energies
from frictiontools.dos import output_dos, calculate_dos_fermi
from frictiontools.jdos import calculate_and_plot_jdos_q0_parallel, calculate_and_plot_jdos_allq_parallel
from frictiontools.friction_tensor import * 
from frictiontools.constants import *
from frictiontools.epc import *
from frictiontools.properties import * 

### Constants --------------------------------------------------------------------------

h = 4.135667696e-3 # Planck's const. in eV*ps
c = 1.23981e-4 # converts between eV and cm^-1
k_B = 8.617333262145e-5 # Boltzmann's const. in eV/K


### IO functions -----------------------------------------------------------------------

def parse_friction_tensor(filepath):
    """
    Parse friction_tensor.out file into a numpy array.
    
    Returns a 2D numpy array where each row is a friction tensor component.
    """
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and empty lines
            if line.startswith('#') or not line:
                continue
            # Parse numerical values
            values = [float(x) for x in line.split()]
            data.append(values)
    
    return np.array(data)

def read_friction_tensor(filepath, structured=False):
    """
    Read friction tensor from file.
    
    Parameters
    ----------
    filepath : str
        Path to friction_tensor.out file
    structured : bool, optional
        If True, return dictionary with 3x3 matrices per atom.
        If False (default), return flattened 2D array.
    
    Returns
    -------
    np.ndarray or dict
        Friction tensor data
    """
    if not structured:
        return parse_friction_tensor(filepath)
    
    tensors = {}
    current_atom = None
    current_tensor = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                if 'n_atom' in line:
                    # Save previous tensor
                    if current_atom is not None and current_tensor:
                        tensors[current_atom] = np.array(current_tensor)
                    # Extract atom index
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'n_atom':
                            current_atom = int(parts[i+1])
                            break
                    current_tensor = []
            elif line and not line.startswith('#'):
                values = [float(x) for x in line.split()]
                current_tensor.append(values)
        
        # Save last tensor
        if current_atom is not None and current_tensor:
            tensors[current_atom] = np.array(current_tensor)
    
    return tensors

def write_friction_tensor(filepath, tensor_data, atom_indices=None, header=True):
    """
    Write friction tensor to file.
    
    Parameters
    ----------
    filepath : str
        Path to output file
    tensor_data : np.ndarray or dict
        If ndarray: 2D array with shape (n_rows, 3)
        If dict: indexed by atom number, values are 3x3 arrays
    atom_indices : list, optional
        Atom indices for labeling (used if tensor_data is ndarray)
    header : bool, optional
        Whether to include header line (default: True)
    """
    with open(filepath, 'w') as f:
        if header:
            f.write("# friction_tensor in 1/ps\n")
        
        if isinstance(tensor_data, dict):
            # Dictionary format with atom indices
            for atom_idx in sorted(tensor_data.keys()):
                tensor = tensor_data[atom_idx]
                for i_cart in range(tensor.shape[0]):
                    f.write(f"# n_atom   {atom_idx} i_cart   {i_cart+1} index   {i_cart+1}\n")
                    row = tensor[i_cart]
                    f.write("  " + "  ".join(f"{val:15.9f}" for val in row) + "\n")
        else:
            # Flat array format
            if atom_indices is None:
                atom_indices = [1] * tensor_data.shape[0]
            
            for i, row in enumerate(tensor_data):
                atom_idx = atom_indices[i] if i < len(atom_indices) else 1
                i_cart = (i % 3) + 1
                f.write(f"# n_atom   {atom_idx} i_cart   {i_cart} index   {i_cart}\n")
                f.write("  " + "  ".join(f"{val:15.9f}" for val in row) + "\n")

### Process phonon-mode data -----------------------------------------------------------

friction_dirname = "/fs/home/lukas/Science/Projects/Indium_on_Si/electronic_friction/InSi_8x2_ELSI_friction/"
friction_aimsout = friction_dirname + "aims.out"
friction_eigvecs = friction_dirname + "friction_eigenvectors.jmol"

""" 
perturbing_frequencies = []

with open(friction_eigvecs, 'r') as file:
    for line in file:

            # Parse the phonon mode frequencies
            if "Mode #" in line:
                period = float(line.split()[5].strip())
                perturbing_frequencies.append(1/period)

perturbing_frequencies = np.array(perturbing_frequencies)
perturbing_energies = perturbing_frequencies * h   # Planck's const. in eV*ps  	

print(perturbing_energies)
ndims = len(perturbing_energies)

"""

perturbing_energies = np.array([0.0])

### Load aims friction tensor output -----------------------------------------------------------

friction_tensor_dirname = friction_dirname + "friction_tensor.out"

aims_fric_tensor = parse_friction_tensor(friction_tensor_dirname)
aims_fric_absolute_trace = np.sum(np.diagonal(np.abs(aims_fric_tensor)))

### Define friction calculation parameters -----------------------------------------------------

N_e = float(sys.argv[1])       # Number of electrons to excite
excited_k_point = 0     # Index of excited electronic k-point (0 = Gamma point)
conduction_temp = float(sys.argv[2])

e_v = -4.60579927       # Valance band energy / chemical potential
e_c = -4.55165517       # Conduction band energy

print(N_e)
print(conduction_temp)

params = FrictionParams(
    friction_aimsout = friction_aimsout,
    friction_dirname = friction_dirname,
    n_spin = 1,
    sigma = 0.6, # Friction broadening width (eV)
    temperature = conduction_temp, # K
    chem_pot = e_v, #eV
    valance_energies = [e_v, e_c], #eV
    num_excited_electrons = N_e,
    excited_k_point = excited_k_point,
    friction_max_energy = 2.4, # eV
    perturbing_energies = perturbing_energies,
    fermi_mode = "Athermal",
    expression = "default",
    delta_order = 1,
    n_jobs = 48, # parallelism
)

system_properties = get_system_properties(params)

print_friction_params(params)
print_system_properties(system_properties)


### Calculate friction tensor --------------------------------------------------------------------------------

start_time = time.time()
friction_tensor = calculate_friction_tensor_parallel(params)


### Process output friction tensor -----------------------------------------------------------------------------

friction_masses = system_properties.friction_masses

relaxation_tensor = np.zeros_like(friction_tensor) # s-1
for p, pe in enumerate(perturbing_energies):
    relaxation_tensor[:,:,p] = mass_weight_tensor(friction_tensor[:,:,p], friction_masses)

inverse_ps_tensor = np.real(np.sum(relaxation_tensor, axis=-1)/1e12)

end_time = time.time()
print(f"Time taken to calculate friction tensor: {end_time-start_time:.2f} seconds")


### Output frictiontools tensor & compare to aims tensor --------------------------------------------------------

print("FictionTools friction tensor (ps-1) :")
print(inverse_ps_tensor)

print("Aims friction tensor (ps-1) :")
print(aims_fric_tensor)

Frob_error = np.sum(inverse_ps_tensor - aims_fric_tensor)
percentage_error = (100*Frob_error)/np.sum(aims_fric_tensor)
print("Frobenious error with true tensor = ", Frob_error, " ps-1")
print("percentage error = ", percentage_error)

write_friction_tensor(f"../InSi_system_data/InSi_calculated_tensors/Athermal_tensor_Ne_{N_e}_T_{conduction_temp}_tensor.txt", inverse_ps_tensor)

### Plotting -----------------------------------------------------------------------------------------------------

#plt.plot(Ne_vals, friction_vals)#, label=f"{broadening_vals[0]}")

#for i in range(1, len(broadening_vals)):
#    plt.plot(broadening_vals, friction_vals[i])#, label=f"{broadening_vals[i]}")

#plt.hlines(aims_fric_absolute_trace, xmin=0.0, xmax=max(Ne_vals), colors="r", linestyles="dashed", label="groundstate trace")

#plt.xlabel("Number of conduction band electrons", fontsize=20)
#plt.ylabel("Friction tensor trace (ps^-1)", fontsize=20)
#plt.title("Friction tensor trace vs conduction band population \n for excitation at the gamma point", fontsize=20)
#plt.legend()
#plt.show()
