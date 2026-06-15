import os
from os.path import join as pathjoin

aims_tensor_path = "/fs/home/lukas/Science/Projects/Indium_on_Si/electronic_friction/InSi_8x2_ELSI_friction/friction_tensor.out"
calculated_tensor_path = "../InSi_system_data/InSi_calculated_tensors/friction_tensors_dfttoolkit_format/Ne_60k_sweep/Ne_200.0_T_60.0_tensor/friction_tensor.out"

atom_indices = []
cartesian_indices = []
other_indices = []

# Parse the fhi-aims friction_tensor.out file to extract header information
with open(aims_tensor_path, 'r') as aims_file:
    for line in aims_file:
        if line.startswith('#'):
            if 'n_atom' in line:
                current_atom = 0
                i_cart = 0
                index = 0

                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'n_atom':
                        current_atom = int(parts[i+1])
                    elif part == 'i_cart':
                        i_cart = int(parts[i+1])
                    elif part == 'index':
                        index = int(parts[i+1])
            
                atom_indices.append(current_atom)
                cartesian_indices.append(i_cart)
                other_indices.append(index)

# Read the calculated tensor file
with open(calculated_tensor_path, 'r') as tensor_file:
    tensor_lines = tensor_file.readlines()

# Replace header lines with corrected values from aims file
label_num = 0
for i, line in enumerate(tensor_lines):
    if line.startswith('#'):
        if 'n_atom' in line:
            tensor_lines[i] = f"# n_atom {atom_indices[label_num]} i_cart   {cartesian_indices[label_num]} index   {other_indices[label_num]}\n"
            label_num += 1



# Write the modified lines back to the file
with open(calculated_tensor_path, 'w') as tensor_file:
    tensor_file.writelines(tensor_lines)

print(f"Successfully updated {calculated_tensor_path}")
print(f"Updated {label_num} header lines with values from {aims_tensor_path}")

