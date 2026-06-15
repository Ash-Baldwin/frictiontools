import numpy as np
from dfttoolkit.vibrations import AimsVibrations
from dfttoolkit.friction import FrictionTensor
import matplotlib.pyplot as plt
import aimstools.Utilities as ut
from copy import deepcopy
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch


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
sigma = 0.1


geom = AimsVibrations("vibrations.in")
vibrations = AimsVibrations("vibrations.in")

friction_tensor = FrictionTensor("../4x2_unit_cell")
friction_tensor_geom = AimsVibrations("../4x2_unit_cell/geometry.in")

matching_indices = get_matching_indices(geom, friction_tensor_geom)

omega2, eigenvectors = vibrations.get_eigenvalues_and_eigenvectors()
omega = vibrations.get_eigenvalues_in_Hz(omega2)
f_inv_cm = vibrations.get_eigenvalues_in_inverse_cm()


# eigenvectors structure: [mode, atom, degree of freedom]
gamma_modes, x_modes = get_modes_at_k_points(eigenvectors)

eigenvectors_gamma = np.zeros((len(gamma_modes), eigenvectors.shape[1], 3))
f_inv_cm_gamma = np.zeros(len(gamma_modes))

for ind, mode in enumerate(gamma_modes):
    eigenvectors_gamma[ind, :, :] = eigenvectors[mode, :, :]
    f_inv_cm_gamma[ind] = f_inv_cm[mode]

eigenvectors_4x2 = np.zeros((len(gamma_modes), len(friction_tensor_geom), 3))

for ind, matching_index in enumerate(matching_indices):
    eigenvectors_4x2[:, ind, :] = eigenvectors_gamma[:, matching_index, :]


# plotting
for vibration_index in range(len(eigenvectors_4x2)):

    eigenvectors = np.array(eigenvectors_4x2)[vibration_index, :, :]
    eigenvectors = np.real(eigenvectors)

    fig = ut.getOneColumnFigure()

    plt.title(
        r"f={:0.2f}".format(np.real(f_inv_cm_gamma[vibration_index]))
        + r" cm$^{-1}$"
    )

    substrate = deepcopy(friction_tensor_geom)
    uncontrained = substrate.remove_unconstrained_atoms()
    # substrate = substrate.get_periodic_replica(
    #     (1, 1, 1), explicit_replications=[[-1, 0, 1, 2], [-1, 0, 1], [0]]
    # )
    substrate.visualise(
        axes=[0, 1],
        hide_axes=True,
        alpha=1,
        linewidth=0.0,
    )

    fade = Rectangle(
        (-50, -50),
        100,
        100,  # x, y, width, height (data coords)
        facecolor="white",
        alpha=0.25,
        zorder=10000,
    )

    ax = plt.gca()
    ax.add_patch(fade)

    molecule = deepcopy(friction_tensor_geom)
    molecule.remove_constrained_atoms()
    # molecule = molecule.get_periodic_replica(
    #     (1, 1, 1), explicit_replications=[[-1, 0, 1, 2], [-1, 0, 1], [0]]
    # )

    molecule.visualise(
        axes=[0, 1],
        min_zorder=100000,
        hide_axes=True,
        linewidth=0.5,
        # xlim=(-5, 15.28764800 * 2 + 5),
        # ylim=(-5, 31.47898400),
    )

    fade = Rectangle(
        (-50, -50),
        100,
        100,  # x, y, width, height (data coords)
        facecolor="white",
        alpha=0.25,
        zorder=150000,
    )

    ax = plt.gca()
    ax.add_patch(fade)

    molecule_in = deepcopy(molecule)
    molecule_in.remove_atoms_by_species("Si")
    # molecule_in = molecule_in.get_periodic_replica(
    #     (1, 1, 1), explicit_replications=[[-1, 0, 1, 2], [-1, 0, 1], [0]]
    # )

    molecule_in.visualise(
        axes=[0, 1],
        min_zorder=200000,
        hide_axes=True,
        linewidth=0.5,
        xlim=(-5, 15.28764800 * 2 + 5),
        ylim=(-5, 31.47898400),
    )

    constrained = np.any(friction_tensor_geom.constrain_relax, axis=1)

    ax = plt.gca()

    for ind in range(len(friction_tensor_geom)):

        if friction_tensor_geom.coords[ind][2] > 14.0:

            pos_1 = friction_tensor_geom.coords[ind][:2]

            ax.quiver(
                pos_1[0],
                pos_1[1],
                eigenvectors[ind][0],
                eigenvectors[ind][1],
                angles="xy",
                scale_units="xy",
                scale=0.2,
                width=0.004,
                headwidth=4,
                headlength=6,
                headaxislength=5,
                zorder=1000000,
            )

    # plt.plot(
    #     (0.0, 15.28764800),
    #     (0.0, 0.0),
    #     "k--",
    #     linewidth=0.5,
    #     zorder=1000000000000,
    # )
    # plt.plot(
    #     (0.0, 15.28764800),
    #     (26.47898400, 26.47898400),
    #     "k--",
    #     linewidth=0.5,
    #     zorder=1000000000000,
    # )
    # plt.plot(
    #     (0.0, 0.0),
    #     (0.0, 26.47898400),
    #     "k--",
    #     linewidth=0.5,
    #     zorder=1000000000000,
    # )
    # plt.plot(
    #     (15.28764800, 15.28764800),
    #     (0.0, 26.47898400),
    #     "k--",
    #     linewidth=0.5,
    #     zorder=1000000000000,
    # )

    # plt.text(-1.75, 24.75, "Si", ha="center", fontsize=7, zorder=1000000000000)
    # plt.text(-1.35, 22.35, "In", ha="center", fontsize=7, zorder=1000000000000)

    fig.savefig(
        "vibration_modes/vibration_{}.png".format(vibration_index), dpi=600
    )
