# The elements H to Kr, one row per element in order of Z:
#   Element(Z, symbol, mass_most_abundant, mass_average, covalent_radius, bragg_slater_radius, bondi_radius)
# Masses in unified atomic mass units, radii in Ångström; `nothing` where the source lists no value.
#
# Sources:
# - mass_most_abundant: the atomic mass of the most abundant isotope (named at the end of each row),
#   from the Atomic Mass Evaluation 2020: W. J. Huang, M. Wang, F. G. Kondev, G. Audi and S. Naimi,
#   Chin. Phys. C 45, 030002 (2021), and M. Wang, W. J. Huang, F. G. Kondev, G. Audi and S. Naimi,
#   Chin. Phys. C 45, 030003 (2021), file mass_1.mas20.txt.
# - mass_average: the CIAAW standard atomic weight (T. Prohaska et al., "Standard atomic weights of
#   the elements 2021 (IUPAC Technical Report)", Pure Appl. Chem. 94, 573 (2022)); for an element
#   whose standard atomic weight is an interval, its conventional value from the same report.
# - covalent_radius: B. Cordero et al., "Covalent radii revisited", Dalton Trans. 2008, 2832;
#   carbon sp³, Mn, Fe and Co low spin.
# - bragg_slater_radius: J. C. Slater, "Atomic Radii in Crystals", J. Chem. Phys. 41, 3199 (1964),
#   which lists no noble gas.
# - bondi_radius: A. Bondi, "van der Waals Volumes and Radii", J. Phys. Chem. 68, 441 (1964).
const ELEMENTS = (
    Element(1, "H", 1.007825031898, 1.008, 0.31, 0.25, 1.2),  # H-1
    Element(2, "He", 4.00260325413, 4.002602, 0.28, nothing, 1.4),  # He-4
    Element(3, "Li", 7.01600343426, 6.94, 1.28, 1.45, 1.82),  # Li-7
    Element(4, "Be", 9.012183062, 9.0121831, 0.96, 1.05, nothing),  # Be-9
    Element(5, "B", 11.009305166, 10.81, 0.84, 0.85, nothing),  # B-11
    Element(6, "C", 12.0, 12.011, 0.76, 0.7, 1.7),  # C-12
    Element(7, "N", 14.00307400425, 14.007, 0.71, 0.65, 1.55),  # N-14
    Element(8, "O", 15.99491461926, 15.999, 0.66, 0.6, 1.52),  # O-16
    Element(9, "F", 18.99840316207, 18.998403162, 0.57, 0.5, 1.47),  # F-19
    Element(10, "Ne", 19.99244017525, 20.1797, 0.58, nothing, 1.54),  # Ne-20
    Element(11, "Na", 22.98976928195, 22.98976928, 1.66, 1.8, 2.27),  # Na-23
    Element(12, "Mg", 23.985041689, 24.305, 1.41, 1.5, 1.73),  # Mg-24
    Element(13, "Al", 26.981538408, 26.9815384, 1.21, 1.25, nothing),  # Al-27
    Element(14, "Si", 27.97692653442, 28.085, 1.11, 1.1, 2.1),  # Si-28
    Element(15, "P", 30.97376199768, 30.973761998, 1.07, 1.0, 1.8),  # P-31
    Element(16, "S", 31.97207117354, 32.06, 1.05, 1.0, 1.8),  # S-32
    Element(17, "Cl", 34.968852694, 35.45, 1.02, 1.0, 1.75),  # Cl-35
    Element(18, "Ar", 39.96238312204, 39.95, 1.06, nothing, 1.88),  # Ar-40
    Element(19, "K", 38.96370648482, 39.0983, 2.03, 2.2, 2.75),  # K-39
    Element(20, "Ca", 39.96259085, 40.078, 1.76, 1.8, nothing),  # Ca-40
    Element(21, "Sc", 44.955907051, 44.955907, 1.7, 1.6, nothing),  # Sc-45
    Element(22, "Ti", 47.947940677, 47.867, 1.6, 1.4, nothing),  # Ti-48
    Element(23, "V", 50.943957664, 50.9415, 1.53, 1.35, nothing),  # V-51
    Element(24, "Cr", 51.940504714, 51.9961, 1.39, 1.4, nothing),  # Cr-52
    Element(25, "Mn", 54.93804304, 54.938043, 1.39, 1.4, nothing),  # Mn-55
    Element(26, "Fe", 55.934935537, 55.845, 1.32, 1.4, nothing),  # Fe-56
    Element(27, "Co", 58.933193524, 58.933194, 1.26, 1.35, nothing),  # Co-59
    Element(28, "Ni", 57.93534165, 58.6934, 1.24, 1.35, 1.63),  # Ni-58
    Element(29, "Cu", 62.929597119, 63.546, 1.32, 1.35, 1.4),  # Cu-63
    Element(30, "Zn", 63.929141776, 65.38, 1.22, 1.35, 1.39),  # Zn-64
    Element(31, "Ga", 68.925573528, 69.723, 1.22, 1.3, 1.87),  # Ga-69
    Element(32, "Ge", 73.92117776, 72.63, 1.2, 1.25, nothing),  # Ge-74
    Element(33, "As", 74.921594562, 74.921595, 1.19, 1.15, 1.85),  # As-75
    Element(34, "Se", 79.916521761, 78.971, 1.2, 1.15, 1.9),  # Se-80
    Element(35, "Br", 78.918337574, 79.904, 1.2, 1.15, 1.85),  # Br-79
    Element(36, "Kr", 83.91149772708, 83.798, 1.16, nothing, 2.02),  # Kr-84
)
