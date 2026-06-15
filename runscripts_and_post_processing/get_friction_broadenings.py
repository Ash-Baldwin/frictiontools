import numpy as np
from dfttoolkit.vibrations import AimsVibrations
from dfttoolkit.friction import FrictionTensor
import matplotlib.pyplot as plt
#from plotutils.formatting import get_presentation_figure
from plot_friction_tensors import read_friction_tensor

def get_matching_indices(geom, geom_match):

    matching_indices = []

    for coord in geom_match.coords:
        residual = geom.coords - coord
        residual = np.sum(abs(residual), axis=1)

        index = np.argmin(residual)
        if residual[index] < 0.1:
            matching_indices.append(index)

    return matching_indices


def get_modes_at_k_points(eigenvectors):
    num_dof_per_cell = eigenvectors.shape[1] // 2
    num_modes = eigenvectors.shape[0]

    gamma_modes = []
    x_modes = []

    for i in range(num_modes):
        mode = eigenvectors[i, :]

        u1 = mode[:num_dof_per_cell].flatten()
        u2 = mode[num_dof_per_cell:].flatten()

        # Calculate alignment score:
        # Score 1: Alignment with the Gamma (in-phase) component
        # We are comparing the component u2 to u1
        gamma_score = np.dot(u1, u2) / (
            np.linalg.norm(u1) * np.linalg.norm(u2)
        )

        # Score 2: Alignment with the X (anti-phase) component
        # We are comparing the component u2 to -u1
        x_score = np.dot(u1, -u2) / (np.linalg.norm(u1) * np.linalg.norm(u2))

        # The value 1 (for Gamma) is ideal for in-phase.
        if np.abs(gamma_score - 1.0) < np.abs(x_score - 1.0):
            # The mode is closer to the Gamma-point symmetry
            gamma_modes.append(i)
        else:
            # The mode is closer to the X-point symmetry
            x_modes.append(i)

    return gamma_modes, x_modes


f_min = 0
f_max = 275
resolution = 10000
sigma = 0.6

excitations = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 50.0, 100.0, 150.0, 200.0]
rotation_mode = []
shear_mode = []

geom = AimsVibrations("../InSi_system_data/vibrations.in")
vibrations = AimsVibrations("../InSi_system_data/vibrations.in")

for (i,Ne) in enumerate(excitations):
    friction_tensor = FrictionTensor(f"../InSi_system_data/InSi_calculated_tensors/friction_tensors_dfttoolkit_format/Ne_60k_sweep/Ne_{Ne}_T_60.0_tensor")
    friction_tensor_geom = AimsVibrations(f"../InSi_system_data/InSi_calculated_tensors/friction_tensors_dfttoolkit_format/Ne_60k_sweep/Ne_{Ne}_T_60.0_tensor/geometry.in")
    print(friction_tensor.friction_tensor)
    matching_indices = get_matching_indices(geom, friction_tensor_geom)
    #print(len(matching_indices))

    omega2, eigenvectors = vibrations.get_eigenvalues_and_eigenvectors()
    omega = vibrations.get_eigenvalues_in_Hz(omega2)
    f_inv_cm = vibrations.get_eigenvalues_in_inverse_cm()


    # eigenvectors structure: [mode, atom, degree of freedom]
    gamma_modes, x_modes = get_modes_at_k_points(eigenvectors)

    eigenvectors_4x2 = np.zeros(
        (eigenvectors.shape[0], len(friction_tensor_geom), 3)
    )

    #print(eigenvectors_4x2.shape)
    #print(eigenvectors.shape)

    for ind, matching_index in enumerate(matching_indices):
        eigenvectors_4x2[:, ind, :] = eigenvectors[:, matching_index, :]

    lifetimes = []
    for mode in range(eigenvectors_4x2.shape[0]):
        """     
        vibration = eigenvectors_4x2[mode]
        vibration /= np.linalg.norm(vibration)
        force = friction_tensor.dot(vibration)
        eta = vibration.dot(force)
        lifetime = 1 / eta 
        """
        
        lifetime = friction_tensor.get_life_time(eigenvectors_4x2[mode].flatten())
        lifetimes.append(lifetime)
        

    lifetimes = np.array(lifetimes)

    sorted_indicies = np.argsort(lifetimes)
    print(sorted_indicies[1:10])

    rotation_mode.append(lifetimes[78])
    shear_mode.append(lifetimes[68])

#------------------------------- Plot rotation & shear mode lifetime bar chart ----------------------------

""" 
fig = plt.figure()
lifetimes = np.sort(lifetimes)

for i in range(0,3):
    plt.bar(i, lifetimes[i], figure=fig, label=f"${i}") #label="rotation mode (X-point)", figure=fig)

#plt.bar(1, lifetimes[1], figure=fig, )#label="shear mode ($\Gamma$-point)", figure=fig)

#plt.xticks([0, 1], ["rotation mode (X-point)", "shear mode ($\Gamma$-point)"], figure=fig)
plt.ylabel("lifetime / ps", figure=fig)

fig.savefig("phonon_mode_lifetimes.pdf") 
"""

#------------------------------- Plot rotation & shear mode lifetime vs excited electrons -----------------

rotation_fig = plt.figure()
print(rotation_mode)
plt.scatter(excitations, rotation_mode, label="rotation mode lifetime (X-point)", marker='X', figure=rotation_fig)

plt.xlabel("Number of $\Gamma$-point electrons excited to conduction band", figure=rotation_fig)
plt.ylabel("lifetime / ps", figure=rotation_fig)

#plt.ylim(rotation_mode[-1]-1e4, rotation_mode[0]+1e4)
#plt.ticklabel_format(axis='y', style='sci')
#plt.rcParams['axes.formatter.useoffset'] = False
plt.title("rotation mode (X-point) lifetime vs excited $\Gamma$-point electrons")
rotation_fig.savefig("../plots/rotation_mode_lifetimes_vs_excitation.pdf", bbox_inches="tight")

shear_fig = plt.figure()

plt.scatter(excitations, shear_mode, label="shear mode lifetime ($\Gamma$-point)", marker='X', figure=shear_fig)

plt.xlabel("Number of $\Gamma$-point electrons excited to conduction band", figure=shear_fig)
plt.ylabel("lifetime / ps", figure=shear_fig)
plt.title("shear mode ($\Gamma$-point) lifetime vs excited $\Gamma$-point electrons")
shear_fig.savefig("../plots/shear_mode_lifetimes_vs_excitation.pdf", bbox_inches="tight")



# for mode in range(eigenvectors_x_4x2.shape[0]):
#     lifetimes = friction_tensor.get_life_time(
#         eigenvectors_x_4x2[mode].flatten()
#     )

#     print(f_inv_cm_x[mode], lifetimes, 1/lifetimes)
