from frictiontools import fermi_integral
from frictiontools import excited_chemical_potentials
from frictiontools import utils
from frictiontools import aims_parse
import matplotlib.pyplot as plt
import numpy as np

ground_state_mu, evs = aims_parse.parse_ev_data("/fs/home/lukas/Science/Projects/Indium_on_Si/electronic_friction/InSi_8x2_ELSI_friction/")

T_v = 60 #K
T_c = 0.1 #K
temperature = [T_v, T_c]
target_electron_deviation = 10.0
k_point = 0

valance_energies = np.array([-4.60572111, -4.55158889]) #valance energies for kpoint 12 - with minimum HOMO-LUMO gap

HOMO = fermi_integral.find_max_occupied_state(evs[k_point,:], ground_state_mu, valance_energies, temperature[0], tolerance=1e-3, fermi_mode="diff")
LUMO = fermi_integral.find_min_unoccupied_state(evs[k_point,:], ground_state_mu, valance_energies, temperature[0],  tolerance=1e-3, fermi_mode="diff")

#valance_energies = np.array([-4.60572111, -4.55158889])
valance_energies = np.array([evs[k_point, HOMO], evs[k_point, LUMO]])
ground_state_mu = valance_energies[0]
print("valance energies = ", valance_energies)

fermi_mode = "excited"

mu_v = excited_chemical_potentials.converge_chemical_potential(
    temperature,
    -1.0*target_electron_deviation,
    evs[k_point,:],
    ground_state_mu,
    fermi_mode,
    band ="valance",
    energy_window=None,
    valance_energies=valance_energies,
    mu_bounds=None,
    tolerance=1e-8
)

mu_c = excited_chemical_potentials.converge_chemical_potential(
    temperature,
    target_electron_deviation,
    evs[k_point,:],
    ground_state_mu,
    fermi_mode,
    band ="conduction",
    energy_window=None,
    valance_energies=valance_energies,
    mu_bounds=None,
    tolerance=1e-8
)

chem_pot = [mu_v, mu_c]
print("chemical potentials = ", chem_pot)
Ne_GS = utils.find_fermi_occupation(evs[k_point,:], ground_state_mu, np.array(temperature)[0], 0, len(evs[k_point,:]))
Ne_EX = utils.find_excited_fermi_occupation(evs[k_point,:], valance_energies, chem_pot, np.array(temperature), 0, len(evs[k_point,:]))
print(Ne_GS)
print(Ne_EX)

nonconservation = Ne_EX - Ne_GS
if abs(nonconservation) >= 1e-3:
    print("non-conservation of electrons = ", nonconservation)

excitation_energy = 0.0
for e in evs[k_point,:]:
    if e < valance_energies[1]:
        excitation_energy += 0.0
    elif e >= valance_energies[1]:
        excitation_energy += (e-ground_state_mu) * utils.excited_fermi_pop(e, valance_energies, chem_pot, temperature)

print("excitation energy = ", excitation_energy, " eV")

occupations = []
gs_occupations = []

min_eigval = 1300
max_eigval = 1400

tight_e_grid = np.arange(ground_state_mu-2.0, ground_state_mu+2.0, 0.01)

for e in tight_e_grid: #evs[k_point, min_eigval:max_eigval]:
    occupations.append(utils.excited_fermi_pop(e, valance_energies, chem_pot, temperature))

for e in tight_e_grid: #evs[k_point, min_eigval:max_eigval]:
    gs_occupations.append(utils.fermi_pop(e, ground_state_mu, temperature[0]))


#plt.plot(evs[k_point, min_eigval:max_eigval]-ground_state_mu, occupations)
#plt.plot(evs[k_point, min_eigval:max_eigval]-ground_state_mu, gs_occupations)

plt.plot(tight_e_grid-ground_state_mu, occupations)
#plt.plot(tight_e_grid-ground_state_mu, gs_occupations)


#plt.vlines(valance_energies[1]-ground_state_mu+0.5, 0.0, 1.0, colors="red", linestyles="dashed")
plt.vlines(valance_energies-ground_state_mu, 0.0, 1.0, colors="green", linestyles="dashed")

plt.savefig("../plots/occupation_spectrum.pdf")
