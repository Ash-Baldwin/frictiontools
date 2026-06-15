import numpy as np
import matplotlib.pyplot as plt

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

### -------------------------------------------------------------------------------------------------

aims_tensor = read_friction_tensor("/fs/home/lukas/Science/Projects/Indium_on_Si/electronic_friction/InSi_8x2_ELSI_friction/friction_tensor.out")

excitations = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
temperatures = [100.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0, 7500.0, 8000.0, 8500.0, 10000.0]
traces = []
magnitudes = []

for (i,Ne) in enumerate(excitations):
    tensor = read_friction_tensor(f"../InSi_system_data/InSi_calculated_tensors/Ne_60k_sweep/Ne_{Ne}_T_60.0_tensor.txt")
    traces.append(np.trace(tensor))
    magnitudes.append(np.sum(abs(tensor)))


plt.scatter(excitations, traces, marker='X')
plt.xlabel("Number of excited electrons")
plt.ylabel("Friction tensor trace / ps^-1")
plt.title("friction tensor trace vs conduction band occupation at 60K")
plt.savefig("../plots/traces_vs_Ne_60k.pdf",bbox_inches='tight') 

""" 
plt.scatter(excitations, magnitudes, marker='X')
plt.xlabel("Number of excited electrons at Gamma point")
plt.ylabel("Friction tensor magnitude / ps^-1")
plt.savefig("abs_magnitudes_vs_excitation.pdf", bbox_inches='tight')
"""