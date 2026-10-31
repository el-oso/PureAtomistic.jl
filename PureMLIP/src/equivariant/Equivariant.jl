"""
    PureMLIP.Equivariant

Textbook real spherical harmonics, Clebsch-Gordan coefficients, Wigner D matrices, and the
conversion layer to the e3nn conventions. It depends on no other part of PureMLIP.
"""
module Equivariant

using LinearAlgebra

export sh_length, sh_index, spherical_harmonics!, spherical_harmonics_and_gradient!
export clebsch_gordan_complex, real_clebsch_gordan, wigner_D
export e3nn_spherical_harmonics!, e3nn_spherical_harmonics_and_gradient!, e3nn_basis_change, e3nn_wigner_3j

include("spherical_harmonics.jl")
include("clebsch_gordan.jl")
include("wigner_d.jl")
include("e3nn.jl")

end
