# The Berendsen barostat with a device potential: the pressure is computed on the device and read
# back as one scalar; `rescale!` scales the cell and the positions on the device with `map!`.

"""
    scale_factor(b::BerendsenBarostat, sys, p::DeviceMACEPotential, dt) -> μ

The Berendsen factor of the CPU method from the pressure `-tr(stress)/3 + 2K/(3V)` of `sys` and
`p.result.stress`, computed on the device (the kinetic energy summed in the tree order of the
device sums) and read back as one scalar, which `readbacks` counts. A singular or non-finite
cell gives a non-finite pressure and the factor's error; a step's force evaluation has already
refused such a cell.
"""
function Barostats.scale_factor(b::BerendsenBarostat, sys::System{T}, p::DeviceMACEPotential{T}, dt::Real)::AbstractFloat where {T <: AbstractFloat}
    _check_motion(sys, p)
    _check_configuration(p, sys.positions, sys.cell)
    return Barostats.scale_factor(b, _device_pressure!(sys, p) * T(EV_PER_A3_IN_GPA), dt)
end

# The pressure in eV/Å³, read back.
function _device_pressure!(sys::System{T}, p::DeviceMACEPotential{T}) where {T}
    _kinetic_sum!(sys, p)
    _launch!(_pressure_kernel!, p.nl.backend, 1, p.scalar, p.kinetic_sum, p.result.stress, sys.cell, T(AMU_IN_EV_FS2_PER_A2 / 2))
    return _read_status!(p, p.host_scalar, p.scalar)[1]
end

@kernel function _pressure_kernel!(out, @Const(ksum), @Const(S), @Const(C), half_amu)
    out[1] = _pressure(half_amu * ksum[1], S, C)
end

# `pressure(sys, stress)` from the kinetic energy `K`, the stress `S` and the cell `C`, with the
# volume |det C| expanded by the first column as in `cell_volume`.
@inline function _pressure(K, S, C)
    tr = S[1, 1] + S[2, 2] + S[3, 3]
    det = C[1, 1] * (C[2, 2] * C[3, 3] - C[2, 3] * C[3, 2]) - C[1, 2] * (C[2, 1] * C[3, 3] - C[2, 3] * C[3, 1]) +
        C[1, 3] * (C[2, 1] * C[3, 2] - C[2, 2] * C[3, 1])
    return -tr / 3 + 2 * K / (3 * abs(det))
end
