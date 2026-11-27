# Oracle: the CODATA 2018 tables. Each constant is exact by definition or the published value, so
# the comparisons are `==`.
@testitem "units: constants" begin
    using QuantumEngine
    U = QuantumEngine.Units
    @test U.BOHR_TO_ANGSTROM == 0.529177210903
    @test U.ANGSTROM_TO_BOHR == 1 / 0.529177210903
    @test U.HARTREE_TO_EV == 27.211386245988
    @test U.HARTREE_TO_KCAL == 627.5094740631
    @test U.HARTREE_TO_JOULE == 4.3597447222071e-18
    @test U.HARTREE_TO_CM1 == 219474.6313632
    @test U.BOHR_TO_METER == 5.29177210903e-11
    @test U.AMU_TO_KG == 1.6605390666e-27
    @test U.ELEMENTARY_CHARGE == 1.602176634e-19
    @test U.BOLTZMANN_K == 1.380649e-23
    @test U.PLANCK_H == 6.62607015e-34
    @test U.AVOGADRO_NA == 6.02214076e23
    @test U.SPEED_OF_LIGHT == 299792458.0
    @test U.GAS_CONSTANT_R == 6.02214076e23 * 1.380649e-23
    # Tolerance (a): the CODATA-derived value 2.541746473 has 10 significant digits.
    @test isapprox(U.AU_TO_DEBYE, 2.541746473; rtol = 1.0e-9)
    for c in (
            U.BOHR_TO_ANGSTROM, U.ANGSTROM_TO_BOHR, U.HARTREE_TO_EV, U.HARTREE_TO_KCAL,
            U.HARTREE_TO_JOULE, U.HARTREE_TO_CM1, U.BOHR_TO_METER, U.AMU_TO_KG, U.ELEMENTARY_CHARGE,
            U.BOLTZMANN_K, U.PLANCK_H, U.AVOGADRO_NA, U.SPEED_OF_LIGHT, U.GAS_CONSTANT_R, U.AU_TO_DEBYE,
        )
        @test c isa Float64
    end
end

# Transcription checks: each value is the literal of the cited source, so the comparisons are `==`.
@testitem "units: element table spot checks" begin
    using QuantumEngine
    using QuantumEngine.Units: covalent_radius, bragg_slater_radius, bondi_radius
    # Standard or conventional atomic weights (CIAAW/IUPAC).
    @test element("H").mass_average == 1.008
    @test element("C").mass_average == 12.011
    @test element("O").mass_average == 15.999
    @test element("Fe").mass_average == 55.845
    @test element("Kr").mass_average == 83.798
    @test element("Rb").mass_average == 85.4678
    @test element("Xe").mass_average == 131.293
    @test element("Au").mass_average == 196.96657
    @test element("U").mass_average == 238.02891
    # Curium has no standard atomic weight: both masses are those of ²⁴⁷Cm, its longest-lived isotope.
    @test element("Cm").mass_average == 247.070352678
    # Most abundant isotope (AME2020): ¹H, ¹²C, ¹⁶O, ⁵⁶Fe, ⁸⁴Kr, ⁸⁵Rb, ¹³²Xe, ¹⁹⁷Au, ²³⁸U, ²⁴⁷Cm.
    @test element("H").mass_most_abundant == 1.007825031898
    @test element("C").mass_most_abundant == 12.0
    @test element("O").mass_most_abundant == 15.99491461926
    @test element("Fe").mass_most_abundant == 55.934935537
    @test element("Kr").mass_most_abundant == 83.91149772708
    @test element("Rb").mass_most_abundant == 84.91178973604
    @test element("Xe").mass_most_abundant == 131.90415508346
    @test element("Au").mass_most_abundant == 196.966570103
    @test element("U").mass_most_abundant == 238.050786936
    @test element("Cm").mass_most_abundant == 247.070352678
    # Covalent radii (Cordero et al. 2008; C sp³, Mn/Fe/Co low spin).
    @test covalent_radius(element("H")) == 0.31
    @test covalent_radius(element("C")) == 0.76
    @test covalent_radius(element("O")) == 0.66
    @test covalent_radius(element("Fe")) == 1.32
    @test covalent_radius(element("Kr")) == 1.16
    @test covalent_radius(element("Rb")) == 2.2
    @test covalent_radius(element("Xe")) == 1.4
    @test covalent_radius(element("Au")) == 1.36
    @test covalent_radius(element("U")) == 1.96
    @test covalent_radius(element("Cm")) == 1.69
    # Bragg–Slater radii (Slater 1964).
    @test bragg_slater_radius(element("H")) == 0.25
    @test bragg_slater_radius(element("C")) == 0.7
    @test bragg_slater_radius(element("O")) == 0.6
    @test bragg_slater_radius(element("Fe")) == 1.4
    @test bragg_slater_radius(element("Rb")) == 2.35
    @test element("Xe").bragg_slater_radius === nothing
    @test bragg_slater_radius(element("Au")) == 1.35
    @test bragg_slater_radius(element("U")) == 1.75
    @test element("Cm").bragg_slater_radius === nothing
    # Van der Waals radii (Bondi 1964).
    @test bondi_radius(element("H")) == 1.2
    @test bondi_radius(element("C")) == 1.7
    @test bondi_radius(element("O")) == 1.52
    @test bondi_radius(element("Kr")) == 2.02
    @test element("Rb").bondi_radius === nothing
    @test bondi_radius(element("Xe")) == 2.16
    @test bondi_radius(element("Au")) == 1.66
    @test bondi_radius(element("U")) == 1.86
    @test element("Cm").bondi_radius === nothing
    # Table invariant: row z holds Z = z, and no symbol repeats.
    @test [element(z).Z for z in 1:96] == 1:96
    @test allunique(element(z).symbol for z in 1:96)
    @test QuantumEngine.Units.nelements() == 96
end

@testitem "units: lookups and errors" begin
    using QuantumEngine
    using QuantumEngine.Units: covalent_radius, bragg_slater_radius, bondi_radius
    @test element("fe") === element("FE") === element(26)
    @test element(SubString("xFe", 2, 3)) === element(26)
    @test element(Int8(8)) === element("O")
    @test_throws "units: unknown element symbol \"Xx\"" element("Xx")
    @test_throws "units: unknown element symbol \"Fé\"" element("Fé")
    @test_throws "units: unknown element symbol \"\\ufeffFe\"" element("\ufeffFe")
    @test_throws "units: unknown element symbol \"Fe\\r\"" element("Fe\r")
    @test_throws "units: no element with Z = 0 in the table (H–Cm)" element(0)
    @test_throws "units: no element with Z = 97 in the table (H–Cm)" element(97)
    @test_throws "units: atomic number must be an integer, got the Bool true" element(true)
    @test_throws "units: Bondi (1964) lists no van der Waals radius for Sc" bondi_radius(element("Sc"))
    @test_throws "units: Slater (1964) lists no Bragg–Slater radius for He" bragg_slater_radius(element("He"))
    @test_throws ArgumentError element("Xx")
end

@testitem "units: symbol_number and ascii_lowercase" begin
    using QuantumEngine.Units: symbol_number, ascii_lowercase
    @test symbol_number("Fe") == symbol_number("fE") == 26
    @test symbol_number("Kr") == 36 && symbol_number("H") == 1
    # The code units `lo:hi` alone are compared.
    @test symbol_number("BqFe", 3, 4) == 26
    @test symbol_number("O@", 1, 1) == 8
    @test symbol_number("Xx") == 0 && symbol_number("") == 0 && symbol_number("Fé") == 0
    @test ascii_lowercase(UInt8('Q')) == UInt8('q')
    @test ascii_lowercase(UInt8('q')) == UInt8('q')
    @test ascii_lowercase(UInt8('@')) == UInt8('@') && ascii_lowercase(UInt8('[')) == UInt8('[')
    @test ascii_lowercase(0xc9) == 0xc9
end

@testitem "units: allocation-free lookup" begin
    using QuantumEngine, StrictModeTest
    # @test_noalloc throws a StrictViolation unless AllocCheck proves the call allocation-free.
    @test (@test_noalloc element("Fe")).Z == 26
    @test (@test_noalloc element(26)).symbol == "Fe"
end
