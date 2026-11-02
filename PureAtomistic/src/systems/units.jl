# Units: length in Å, energy in eV, time in fs, mass in amu, temperature in K.

# SI 2019 defining constants (exact): the elementary charge in C (the joule per electronvolt)
# and the Boltzmann constant in J/K.
const _E_CHARGE = 1.602176634e-19
const _K_BOLTZMANN_J = 1.380649e-23
# Atomic mass unit in kg, CODATA 2018.
const _AMU_KG = 1.6605390666e-27

"""
    KB_EV_PER_K

Boltzmann constant in eV/K: `1.380649e-23 J/K / 1.602176634e-19 J/eV = 8.617333262e-5`
(SI 2019 defining values).
"""
const KB_EV_PER_K = _K_BOLTZMANN_J / _E_CHARGE

"""
    AMU_IN_EV_FS2_PER_A2

One atomic mass unit expressed in eV·fs²/Å²: `1.66053906660e-27 kg / (1.602176634e-19 J/eV ·
1e-10 s²/m² per fs²/Å²) = 103.6426965` (CODATA 2018 for the atomic mass unit). The kinetic energy
of a mass `m` in amu moving at `v` Å/fs is `AMU_IN_EV_FS2_PER_A2 · m v² / 2` eV.
"""
const AMU_IN_EV_FS2_PER_A2 = _AMU_KG / (_E_CHARGE * 1.0e-10)

"""
    EV_PER_A3_IN_GPA

One eV/Å³ in GPa: `1.602176634e-19 J / 1e-30 m³ / 1e9 = 160.2176634`.
"""
const EV_PER_A3_IN_GPA = _E_CHARGE / 1.0e-30 / 1.0e9
