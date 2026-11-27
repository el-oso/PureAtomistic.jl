# The elements H to Cm, one row per element in order of Z:
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
# - Tc, Pm, Po, At, Rn, Fr, Ra, Ac, Np, Pu, Am and Cm have no standard atomic weight in that report:
#   both masses are the AME2020 mass of the longest-lived isotope (half-lives from NUBASE2020:
#   F. G. Kondev et al., Chin. Phys. C 45, 030001 (2021)), marked "longest-lived" in the row.
# - covalent_radius: B. Cordero et al., "Covalent radii revisited", Dalton Trans. 2008, 2832;
#   carbon sp³, Mn, Fe and Co low spin.
# - bragg_slater_radius: J. C. Slater, "Atomic Radii in Crystals", J. Chem. Phys. 41, 3199 (1964),
#   which lists no noble gas and no At, Fr or Cm.
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
    Element(37, "Rb", 84.91178973604, 85.4678, 2.2, 2.35, nothing),  # Rb-85
    Element(38, "Sr", 87.905612253, 87.62, 1.95, 2.0, nothing),  # Sr-88
    Element(39, "Y", 88.905838156, 88.905838, 1.9, 1.8, nothing),  # Y-89
    Element(40, "Zr", 89.904698755, 91.224, 1.75, 1.55, nothing),  # Zr-90
    Element(41, "Nb", 92.90637317, 92.90637, 1.64, 1.45, nothing),  # Nb-93
    Element(42, "Mo", 97.905403609, 95.95, 1.54, 1.45, nothing),  # Mo-98
    Element(43, "Tc", 96.90636072, 96.90636072, 1.47, 1.35, nothing),  # Tc-97 (longest-lived)
    Element(44, "Ru", 101.904340312, 101.07, 1.46, 1.3, nothing),  # Ru-102
    Element(45, "Rh", 102.905494081, 102.90549, 1.42, 1.35, nothing),  # Rh-103
    Element(46, "Pd", 105.903480287, 106.42, 1.39, 1.4, 1.63),  # Pd-106
    Element(47, "Ag", 106.905091509, 107.8682, 1.45, 1.6, 1.72),  # Ag-107
    Element(48, "Cd", 113.903364998, 112.414, 1.44, 1.55, 1.58),  # Cd-114
    Element(49, "In", 114.903878772, 114.818, 1.42, 1.55, 1.93),  # In-115
    Element(50, "Sn", 119.902202557, 118.71, 1.39, 1.45, 2.17),  # Sn-120
    Element(51, "Sb", 120.903811353, 121.76, 1.39, 1.45, nothing),  # Sb-121
    Element(52, "Te", 129.906222745, 127.6, 1.38, 1.4, 2.06),  # Te-130
    Element(53, "I", 126.904472592, 126.90447, 1.39, 1.4, 1.98),  # I-127
    Element(54, "Xe", 131.90415508346, 131.293, 1.4, nothing, 2.16),  # Xe-132
    Element(55, "Cs", 132.905451958, 132.90545196, 2.44, 2.6, nothing),  # Cs-133
    Element(56, "Ba", 137.905247059, 137.327, 2.15, 2.15, nothing),  # Ba-138
    Element(57, "La", 138.906362927, 138.90547, 2.07, 1.95, nothing),  # La-139
    Element(58, "Ce", 139.905448433, 140.116, 2.04, 1.85, nothing),  # Ce-140
    Element(59, "Pr", 140.907659604, 140.90766, 2.03, 1.85, nothing),  # Pr-141
    Element(60, "Nd", 141.907728824, 144.242, 2.01, 1.85, nothing),  # Nd-142
    Element(61, "Pm", 144.912755748, 144.912755748, 1.99, 1.85, nothing),  # Pm-145 (longest-lived)
    Element(62, "Sm", 151.919738646, 150.36, 1.98, 1.85, nothing),  # Sm-152
    Element(63, "Eu", 152.921236789, 151.964, 1.98, 1.85, nothing),  # Eu-153
    Element(64, "Gd", 157.9241112, 157.25, 1.96, 1.8, nothing),  # Gd-158
    Element(65, "Tb", 158.925353707, 158.925354, 1.94, 1.75, nothing),  # Tb-159
    Element(66, "Dy", 163.929180819, 162.5, 1.92, 1.75, nothing),  # Dy-164
    Element(67, "Ho", 164.930329116, 164.930329, 1.92, 1.75, nothing),  # Ho-165
    Element(68, "Er", 165.930301067, 167.259, 1.89, 1.75, nothing),  # Er-166
    Element(69, "Tm", 168.934218956, 168.934219, 1.9, 1.75, nothing),  # Tm-169
    Element(70, "Yb", 173.938867545, 173.045, 1.87, 1.75, nothing),  # Yb-174
    Element(71, "Lu", 174.940777211, 174.9668, 1.87, 1.75, nothing),  # Lu-175
    Element(72, "Hf", 179.946559537, 178.486, 1.75, 1.55, nothing),  # Hf-180
    Element(73, "Ta", 180.947998528, 180.94788, 1.7, 1.45, nothing),  # Ta-181
    Element(74, "W", 183.95093318, 183.84, 1.62, 1.35, nothing),  # W-184
    Element(75, "Re", 186.955752217, 186.207, 1.51, 1.35, nothing),  # Re-187
    Element(76, "Os", 191.961478765, 190.23, 1.44, 1.3, nothing),  # Os-192
    Element(77, "Ir", 192.962923753, 192.217, 1.41, 1.35, nothing),  # Ir-193
    Element(78, "Pt", 194.964794325, 195.084, 1.36, 1.35, 1.75),  # Pt-195
    Element(79, "Au", 196.966570103, 196.96657, 1.36, 1.35, 1.66),  # Au-197
    Element(80, "Hg", 201.970643604, 200.592, 1.32, 1.5, 1.55),  # Hg-202
    Element(81, "Tl", 204.974427318, 204.38, 1.45, 1.9, 1.96),  # Tl-205
    Element(82, "Pb", 207.976652005, 207.2, 1.46, 1.8, 2.02),  # Pb-208
    Element(83, "Bi", 208.980398599, 208.9804, 1.48, 1.6, nothing),  # Bi-209
    Element(84, "Po", 208.982430361, 208.982430361, 1.4, 1.9, nothing),  # Po-209 (longest-lived)
    Element(85, "At", 209.987147423, 209.987147423, 1.5, nothing, nothing),  # At-210 (longest-lived)
    Element(86, "Rn", 222.017576017, 222.017576017, 1.5, nothing, nothing),  # Rn-222 (longest-lived)
    Element(87, "Fr", 223.019734241, 223.019734241, 2.6, nothing, nothing),  # Fr-223 (longest-lived)
    Element(88, "Ra", 226.025408186, 226.025408186, 2.21, 2.15, nothing),  # Ra-226 (longest-lived)
    Element(89, "Ac", 227.027750594, 227.027750594, 2.15, 1.95, nothing),  # Ac-227 (longest-lived)
    Element(90, "Th", 232.038053606, 232.0377, 2.06, 1.8, nothing),  # Th-232
    Element(91, "Pa", 231.0358825, 231.03588, 2.0, 1.8, nothing),  # Pa-231
    Element(92, "U", 238.050786936, 238.02891, 1.96, 1.75, 1.86),  # U-238
    Element(93, "Np", 237.04817164, 237.04817164, 1.9, 1.75, nothing),  # Np-237 (longest-lived)
    Element(94, "Pu", 244.064204401, 244.064204401, 1.87, 1.75, nothing),  # Pu-244 (longest-lived)
    Element(95, "Am", 243.061379889, 243.061379889, 1.8, 1.75, nothing),  # Am-243 (longest-lived)
    Element(96, "Cm", 247.070352678, 247.070352678, 1.69, nothing, nothing),  # Cm-247 (longest-lived)
)
