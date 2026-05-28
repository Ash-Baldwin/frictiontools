import numpy as np

def parse_normal_modes(filename, ndim):
    
    lines = open(filename).readlines() #ase vib.py output

    modes = np.zeros([ndim,ndim])
    for i, line in enumerate(lines):
        if 'Zero-point energy' in line:
            j = i+1
            for a in range(ndim):
                for b in range(ndim):
                    modes[a,b] = \
                            float(lines[j].split()[b])
                j += 1

    for i in range(ndim):
        modes[i,:]/=np.linalg.norm(modes[i,:])

    return modes

def parse_perturbing_energies(file_path):
    """
    Parses the perturbing energies of the modes from a file and returns them in eV.

    Parameters:
        file_path (str): Path to the file containing the mode data.

    Returns:
        numpy.ndarray: Array of perturbing energies in eV.
    """
    energies_eV = []

    with open(file_path, 'r') as file:
        mode_section = False  # Toggle to detect the mode energy section
        for line in file:
            if "cm^-1" in line:  # Identify the header and separator
                mode_section = not mode_section
                continue

            if mode_section:
                parts = line.split()
                if len(parts) >= 5:  # Ensure the line has enough columns
                    try:
                        energy_inv_cm = float(parts[4].replace('i', ''))  # Ignore 'i' for imaginary values
                        
                        energies_eV.append(energy_inv_cm*1.23981e-4)
                    except ValueError:
                        continue  # Skip lines that don't parse correctly

    # Convert from meV to eV (1 eV = 1000 meV)
    #energies_eV = np.array(energies_meV) / 1000
    return energies_eV