"""
    QuantumEngine.Units

Physical constants (CODATA 2018) and the element table for H to Cm. Every conversion in
QuantumEngine uses these constants. Lengths in the table are in Ångström, masses in unified atomic
mass units.
"""
module Units

using StrictMode: @strict_policy

# CODATA 2018 recommended values; the SI defining constants are exact.
"The Bohr radius in Å: Å per bohr."
const BOHR_TO_ANGSTROM = 0.529177210903
"Bohr per Å, `1 / BOHR_TO_ANGSTROM`."
const ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM
"The hartree in eV."
const HARTREE_TO_EV = 27.211386245988
"The hartree in kcal/mol."
const HARTREE_TO_KCAL = 627.5094740631
"The hartree in J."
const HARTREE_TO_JOULE = 4.3597447222071e-18
"The hartree in cm⁻¹ (as a wavenumber)."
const HARTREE_TO_CM1 = 219474.6313632

"The Bohr radius in m."
const BOHR_TO_METER = 5.29177210903e-11
"The unified atomic mass unit in kg."
const AMU_TO_KG = 1.6605390666e-27
"The elementary charge in C (exact)."
const ELEMENTARY_CHARGE = 1.602176634e-19
"The Boltzmann constant in J/K (exact)."
const BOLTZMANN_K = 1.380649e-23
"The Planck constant in J s (exact)."
const PLANCK_H = 6.62607015e-34
"The Avogadro constant in mol⁻¹ (exact)."
const AVOGADRO_NA = 6.02214076e23
"The speed of light in m/s (exact)."
const SPEED_OF_LIGHT = 299792458.0

"The molar gas constant in J/(mol K), `AVOGADRO_NA * BOLTZMANN_K`."
const GAS_CONSTANT_R = AVOGADRO_NA * BOLTZMANN_K
# One debye is 1e-21 / c coulomb meter, and the atomic unit of dipole moment e·a₀.
"The atomic unit of dipole moment, e·a₀, in debye."
const AU_TO_DEBYE = ELEMENTARY_CHARGE * BOHR_TO_METER * SPEED_OF_LIGHT * 1.0e21

"""
    Element

One row of the element table: the atomic number `Z`, the `symbol`, the mass of the most abundant
isotope `mass_most_abundant` and the standard (or conventional) atomic weight `mass_average`, both
in amu, and the `covalent_radius`, `bragg_slater_radius` and `bondi_radius` in Å. For an element
with no standard atomic weight (Tc, Pm, Po to Ac, Np to Cm), both masses are that of its
longest-lived isotope. A radius is `nothing` where its source lists no value;
[`bragg_slater_radius`](@ref) and [`bondi_radius`](@ref) throw for those.
"""
struct Element
    Z::Int
    symbol::String
    mass_most_abundant::Float64
    mass_average::Float64
    covalent_radius::Float64
    bragg_slater_radius::Union{Nothing, Float64}
    bondi_radius::Union{Nothing, Float64}
end

include("elements.jl")

"""
    nelements() -> Int

The number of elements in the table (H to Cm).
"""
nelements() = 96

"""
    element(Z::Integer) -> Element
    element(symbol::AbstractString) -> Element

The table row of atomic number `Z`, or of the element symbol `symbol`, which is matched without
regard to case (`"fe"`, `"Fe"` and `"FE"` are iron). Throws an `ArgumentError` for a `Z` outside
1 to 96, a `Bool` `Z`, or an unknown symbol.
"""
function element(Z::Integer)
    1 <= Z <= nelements() || throw(ArgumentError(lazy"units: no element with Z = $(Z) in the table (H–Cm)"))
    return ELEMENTS[Int(Z)]
end

# `Bool <: Integer`, but a Bool is not an atomic number (Base likewise rejects a Bool index).
element(Z::Bool) = throw(ArgumentError(lazy"units: atomic number must be an integer, got the Bool $(Z)"))

function element(symbol::AbstractString)
    z = symbol_number(symbol)
    z == 0 && throw(ArgumentError(lazy"units: unknown element symbol $(repr(symbol))"))
    return ELEMENTS[z]
end

"""
    ascii_lowercase(c::UInt8) -> UInt8

The code unit `c` with an ASCII capital letter lowercased; every other code unit, a non-ASCII one
included, is returned unchanged.
"""
ascii_lowercase(c::UInt8) = UInt8('A') <= c <= UInt8('Z') ? c + (UInt8('a') - UInt8('A')) : c

"""
    symbol_number(s::AbstractString, lo::Int = 1, hi::Int = ncodeunits(s)) -> Int

The atomic number of the element whose symbol the code units `lo:hi` of `s` spell without regard
to ASCII case, or 0 if none does. The comparison is code unit by code unit, so no lowercased copy
is made, and a non-ASCII code unit matches no symbol's letter.
"""
function symbol_number(s::AbstractString, lo::Int = 1, hi::Int = ncodeunits(s))
    for z in 1:nelements()
        ref = ELEMENTS[z].symbol
        hi - lo + 1 == ncodeunits(ref) || continue
        same = true
        for k in 1:ncodeunits(ref)
            if ascii_lowercase(codeunit(s, lo + k - 1)) != ascii_lowercase(codeunit(ref, k))
                same = false
                break
            end
        end
        same && return z
    end
    return 0
end

"""
    covalent_radius(e::Element) -> Float64

The single-bond covalent radius of Cordero et al. (2008), in Å.
"""
covalent_radius(e::Element) = e.covalent_radius

"""
    bragg_slater_radius(e::Element) -> Float64

The atomic radius of Slater (1964), in Å. Throws an `ArgumentError` for the noble gases, At, Fr and
Cm, which Slater does not list.
"""
function bragg_slater_radius(e::Element)
    r = e.bragg_slater_radius
    isnothing(r) && throw(ArgumentError(lazy"units: Slater (1964) lists no Bragg–Slater radius for $(e.symbol)"))
    return r
end

"""
    bondi_radius(e::Element) -> Float64

The van der Waals radius of Bondi (1964), in Å. Throws an `ArgumentError` for the elements Bondi
does not list.
"""
function bondi_radius(e::Element)
    r = e.bondi_radius
    isnothing(r) && throw(ArgumentError(lazy"units: Bondi (1964) lists no van der Waals radius for $(e.symbol)"))
    return r
end

public BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, HARTREE_TO_EV, HARTREE_TO_KCAL, HARTREE_TO_JOULE,
    HARTREE_TO_CM1, BOHR_TO_METER, AMU_TO_KG, ELEMENTARY_CHARGE, BOLTZMANN_K, PLANCK_H,
    AVOGADRO_NA, SPEED_OF_LIGHT, GAS_CONSTANT_R, AU_TO_DEBYE,
    Element, element, covalent_radius, bragg_slater_radius, bondi_radius, nelements, ascii_lowercase,
    symbol_number

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
