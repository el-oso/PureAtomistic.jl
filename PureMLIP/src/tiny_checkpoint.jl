# A tiny ScaleShiftMACE written as a torch checkpoint in memory: a STORED ZIP64 archive holding a
# protocol-2 pickle and two storages, laid out as `torch.save` writes it. It has the supported
# architecture (see `MACEModel`) with C = 2 channels, the elements 1 and 29, a radial MLP of
# widths [4, 4, 4], 10 Bessel functions, and one nonzero entry in each U matrix. Every
# parameter element at position k of the Float64 storage is 0.1 sin(k). It uses no file or
# network access, so it can drive the precompile workload through the real reader and model.

const _TINY_ELEMENTS = [1, 29]
const _TINY_C = 2
const _TINY_NB = 10
const _TINY_R_MAX = 6.0
const _TINY_SILU_CST = 1.679176792398942
const _TINY_COVALENT_RADII = [
    0.2, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11,
    1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.7, 1.6, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32,
]

# Little-endian bytes of `x`, appended to `out`.
function _put!(out::Vector{UInt8}, x::Unsigned)
    for b in 0:(sizeof(x) - 1)
        push!(out, (x >> (8b)) % UInt8)
    end
    return out
end

function _put!(out::Vector{UInt8}, x::Unsigned, rest::Unsigned...)
    _put!(out, x)
    return _put!(out, rest...)
end

# ZIP64 archive of uncompressed entries, as torch writes it: classic local headers, central
# records with 0xFFFFFFFF sentinels and a ZIP64 extra field, then the ZIP64 end records.
function _tiny_zip(entries::Vector{Pair{String, Vector{UInt8}}})
    out = UInt8[]
    central = UInt8[]
    for (name, data) in entries
        off = UInt64(length(out))
        crc = crc32(data)
        nm = codeunits(name)
        sz = UInt64(length(data))
        _put!(out, 0x04034b50, 0x000a, 0x0000, 0x0000, 0x0000, 0x0021, crc, UInt32(sz), UInt32(sz), UInt16(length(nm)), 0x0000)
        append!(out, nm)
        append!(out, data)
        _put!(central, 0x02014b50, 0x002d, 0x000a, 0x0000, 0x0000, 0x0000, 0x0021, crc, 0xffffffff, 0xffffffff, UInt16(length(nm)), UInt16(28))
        _put!(central, 0x0000, 0x0000, 0x0000, 0x00000000, 0xffffffff)
        append!(central, nm)
        _put!(central, 0x0001, UInt16(24), sz, sz, off)
    end
    cd_offset = UInt64(length(out))
    append!(out, central)
    n = UInt64(length(entries))
    z64_offset = UInt64(length(out))
    _put!(out, 0x06064b50, UInt64(44), 0x002d, 0x002d, 0x00000000, 0x00000000, n, n, UInt64(length(central)), cd_offset)
    _put!(out, 0x07064b50, 0x00000000, z64_offset, 0x00000001)
    _put!(out, 0x06054b50, 0x0000, 0x0000, 0xffff, 0xffff, 0xffffffff, 0xffffffff, 0x0000)
    return out
end

# Protocol-2 pickle fragments; each is the bytes that leave one value on the stack.
_pk_cat(xs::Vector{UInt8}...) = reduce(vcat, xs; init = UInt8[])
_pk_str(s::String) = UInt8[UInt8('X'), _put!(UInt8[], UInt32(ncodeunits(s)))..., codeunits(s)...]
_pk_int(n::Integer) = UInt8[UInt8('J'), _put!(UInt8[], unsigned(Int32(n)))...]
_pk_bool(b::Bool) = UInt8[b ? 0x88 : 0x89]
_pk_float(x::Real) = UInt8[UInt8('G'), reverse(_put!(UInt8[], reinterpret(UInt64, Float64(x))))...]
_pk_global(m::String, n::String) = UInt8[UInt8('c'), codeunits(m)..., UInt8('\n'), codeunits(n)..., UInt8('\n')]
_pk_tuple(xs::Vector{UInt8}...) = UInt8[UInt8('('), _pk_cat(xs...)..., UInt8('t')]
_pk_list(xs::Vector{UInt8}...) = UInt8[UInt8(']'), UInt8('('), _pk_cat(xs...)..., UInt8('e')]
_pk_dict(kvs::Pair{String, Vector{UInt8}}...) =
    UInt8[UInt8('}'), UInt8('('), _pk_cat([vcat(_pk_str(k), v) for (k, v) in kvs]...)..., UInt8('u')]
_pk_call(callable::Vector{UInt8}, args::Vector{UInt8}) = UInt8[callable..., args..., UInt8('R')]
_pk_ints(xs::Vector{Int}) = _pk_tuple([_pk_int(x) for x in xs]...)

const _PK_ODICT = UInt8[_pk_global("collections", "OrderedDict")..., 0x29, UInt8('R')]

# A module object: `cls()` followed by BUILD with its `__dict__`.
function _tiny_module(
        cls::Tuple{String, String};
        params = Pair{String, Vector{UInt8}}[], buffers = Pair{String, Vector{UInt8}}[],
        modules = Pair{String, Vector{UInt8}}[], attrs = Pair{String, Vector{UInt8}}[]
    )
    state = Pair{String, Vector{UInt8}}[
        "_parameters" => _pk_dict(params...), "_buffers" => _pk_dict(buffers...), "_modules" => _pk_dict(modules...),
    ]
    append!(state, attrs)
    return UInt8[_pk_call(_pk_global(cls...), UInt8[0x29])..., _pk_dict(state...)..., UInt8('b')]
end

_container() = ("torch.nn.modules.container", "ModuleList")
_modulelist(xs::Vector{UInt8}...) = _tiny_module(_container(); modules = Pair{String, Vector{UInt8}}[string(i - 1) => x for (i, x) in enumerate(xs)])

# The two storages of the archive. `n_f64` and `n_i64` are the element counts that the
# persistent ids record; they are known only after a first build.
mutable struct _TinyStorages
    f64::Vector{Float64}
    i64::Vector{Int64}
    n_f64::Int
    n_i64::Int
end

_row_major_strides(shape::Vector{Int}) = [prod(shape[(d + 1):end]; init = 1) for d in eachindex(shape)]

function _tiny_tensor(cls::String, key::String, n::Int, offset::Int, shape::Vector{Int})
    storage = UInt8[_pk_tuple(_pk_str("storage"), _pk_global("torch", cls), _pk_str(key), _pk_str("cpu"), _pk_int(n))..., UInt8('Q')]
    return _pk_call(
        _pk_global("torch._utils", "_rebuild_tensor_v2"),
        _pk_tuple(storage, _pk_int(offset), _pk_ints(shape), _pk_ints(_row_major_strides(shape)), _pk_bool(false), _PK_ODICT)
    )
end

# A Float64 tensor of the given row-major values.
function _tiny_f64!(st::_TinyStorages, shape::Vector{Int}, values::AbstractVector{Float64})
    prod(shape; init = 1) == length(values) || throw(ArgumentError(lazy"tiny checkpoint: shape $shape does not hold $(length(values)) values"))
    offset = length(st.f64)
    append!(st.f64, values)
    return _tiny_tensor("DoubleStorage", "0", st.n_f64, offset, shape)
end

function _tiny_i64!(st::_TinyStorages, shape::Vector{Int}, values::Vector{Int})
    prod(shape; init = 1) == length(values) || throw(ArgumentError(lazy"tiny checkpoint: shape $shape does not hold $(length(values)) values"))
    offset = length(st.i64)
    append!(st.i64, values)
    return _tiny_tensor("LongStorage", "1", st.n_i64, offset, shape)
end

_scalar!(st::_TinyStorages, x::Real) = _tiny_f64!(st, Int[], [Float64(x)])

# A parameter whose element at position k of the Float64 storage is 0.1 sin(k).
function _weight!(st::_TinyStorages, shape::Vector{Int})
    first = length(st.f64) + 1
    t = _tiny_f64!(st, shape, [0.1 * sin(k) for k in first:(first + prod(shape) - 1)])
    return _pk_call(_pk_global("torch._utils", "_rebuild_parameter"), _pk_tuple(t, _pk_bool(true), _PK_ODICT))
end

_buffers(st, xs::Pair{String, <:Real}...) = Pair{String, Vector{UInt8}}[k => _scalar!(st, v) for (k, v) in xs]

const _Irreps = Vector{NTuple{3, Int}}   # (multiplicity, l, parity)

function _pk_irreps(terms::_Irreps)
    mulir(m, l, p) = _pk_call(
        _pk_global("e3nn.o3._irreps", "_MulIr"),
        _pk_tuple(_pk_tuple(_pk_int(m), _pk_call(_pk_global("e3nn.o3._irreps", "Irrep"), _pk_tuple(_pk_tuple(_pk_int(l), _pk_int(p))))))
    )
    return _pk_call(_pk_global("e3nn.o3._irreps", "Irreps"), _pk_tuple(_pk_tuple([mulir(t...) for t in terms]...)))
end

_parity(l) = iseven(l) ? 1 : -1

# One term of multiplicity `mul` for each degree in `ls`.
_terms(mul::Int, ls) = _Irreps([(mul, l, _parity(l)) for l in ls])

# e3nn `Linear` over `ins = (input term, output term)` pairs, 0-based. The path weight of an
# instruction is 1/sqrt of the input multiplicities that reach its output term.
function _tiny_linear!(st::_TinyStorages, irin::_Irreps, irout::_Irreps, ins::Vector{Tuple{Int, Int}})
    fan = zeros(Int, length(irout))
    for (i, o) in ins
        fan[o + 1] += irin[i + 1][1]
    end
    instructions = [
        _pk_call(
            _pk_global("e3nn.o3._linear", "Instruction"),
            _pk_tuple(_pk_int(i), _pk_int(o), _pk_ints([irin[i + 1][1], irout[o + 1][1]]), _pk_float(1 / sqrt(fan[o + 1])))
        ) for (i, o) in ins
    ]
    n = sum(irin[i + 1][1] * irout[o + 1][1] for (i, o) in ins)
    return _tiny_module(
        ("e3nn.o3._linear", "Linear");
        params = ["weight" => _weight!(st, [n])],
        attrs = ["irreps_in" => _pk_irreps(irin), "irreps_out" => _pk_irreps(irout), "instructions" => _pk_list(instructions...)]
    )
end

# e3nn tensor product with instructions `(in1, in2, out, mode, path weight, path shape)`, 0-based.
# A product with `weighted` set owns a flat weight parameter.
function _tiny_tp!(
        st::_TinyStorages, cls::Tuple{String, String}, in1::_Irreps, in2::_Irreps, out::_Irreps,
        ins::Vector{Tuple{Int, Int, Int, String, Float64, Vector{Int}}}; weighted::Bool
    )
    instructions = [
        _pk_call(
            _pk_global("e3nn.o3._tensor_product._instruction", "Instruction"),
            _pk_tuple(_pk_int(a), _pk_int(b), _pk_int(c), _pk_str(mode), _pk_bool(true), _pk_float(pw), _pk_ints(shape))
        ) for (a, b, c, mode, pw, shape) in ins
    ]
    n = sum(prod(shape) for (_, _, _, _, _, shape) in ins)
    return _tiny_module(
        cls;
        params = weighted ? ["weight" => _weight!(st, [n])] : Pair{String, Vector{UInt8}}[],
        attrs = [
            "irreps_in1" => _pk_irreps(in1), "irreps_in2" => _pk_irreps(in2), "irreps_out" => _pk_irreps(out),
            "instructions" => _pk_list(instructions...),
        ]
    )
end

_silu() = _tiny_module(
    ("e3nn.math._normalize_activation", "normalize2mom");
    attrs = ["f" => _pk_global("torch.nn.functional", "silu"), "cst" => _pk_float(_TINY_SILU_CST)]
)

# FullyConnectedNet of widths `widths`, without biases; layer j < 3 owns an activation.
function _tiny_radial_mlp!(st::_TinyStorages, widths::Vector{Int})
    layers = Pair{String, Vector{UInt8}}[]
    for j in 1:4
        modules = j < 4 ? ["act" => _silu()] : Pair{String, Vector{UInt8}}[]
        push!(layers, string("layer", j - 1) => _tiny_module(("e3nn.nn._fc", "_Layer"); params = ["weight" => _weight!(st, [widths[j], widths[j + 1]])], modules))
    end
    return _tiny_module(("e3nn.nn._fc", "FullyConnectedNet"); modules = layers)
end

# U matrix of the given shape with a single entry 1 at 0-based coordinates `at`.
function _tiny_u!(st::_TinyStorages, shape::Vector{Int}, at::Vector{Int})
    u = zeros(prod(shape))
    u[sum((at[d] * prod(shape[(d + 1):end]; init = 1) for d in eachindex(shape))) + 1] = 1.0
    return _tiny_f64!(st, shape, u)
end

# Contraction of correlation 3 for output degree `l`. Its coupling paths are each one entry:
# the first scalar component, or, for `l = 1`, one component of the first vector slot.
function _tiny_contraction!(st::_TinyStorages, l::Int)
    L, ns, C = 16, length(_TINY_ELEMENTS), _TINY_C
    lead = l == 0 ? Int[] : [2l + 1]
    z = l == 0 ? Int[] : [0]
    w = l == 0 ? 0 : 1
    weights = _tiny_module(
        _container();
        params = ["0" => _weight!(st, [ns, 1, C]), "1" => _weight!(st, [ns, 1, C])]
    )
    return _tiny_module(
        ("mace.modules.symmetric_contraction", "Contraction");
        params = ["weights_max" => _weight!(st, [ns, 1, C])],
        buffers = [
            "U_matrix_1" => _tiny_u!(st, [lead; L; 1], [z; w; 0]),
            "U_matrix_2" => _tiny_u!(st, [lead; L; L; 1], [z; w; 0; 0]),
            "U_matrix_3" => _tiny_u!(st, [lead; L; L; L; 1], [z; w; 0; 0; 0]),
        ],
        modules = ["weights" => weights]
    )
end

function _tiny_product!(st::_TinyStorages, ls::Vector{Int}, use_sc::Bool, linear::Vector{UInt8})
    C = _TINY_C
    contractions = _modulelist([_tiny_contraction!(st, l) for l in ls]...)
    sc = _tiny_module(
        ("mace.modules.symmetric_contraction", "SymmetricContraction");
        modules = ["contractions" => contractions]
    )
    return _tiny_module(
        ("mace.modules.blocks", "EquivariantProductBasisBlock");
        modules = ["symmetric_contractions" => sc, "linear" => linear],
        attrs = ["use_sc" => _pk_bool(use_sc)]
    )
end

# Instructions of the conv_tp of the second interaction: a scalar-and-vector input coupled to
# the four edge-attribute degrees, one output slot per path.
const _TINY_CONV1 = [
    (0, 0, 0, 1.0), (1, 1, 1, 1.0), (0, 1, 2, sqrt(3.0)), (1, 0, 3, sqrt(3.0)), (1, 2, 4, sqrt(3.0)),
    (0, 2, 5, sqrt(5.0)), (1, 1, 6, sqrt(5.0)), (1, 3, 7, sqrt(5.0)), (0, 3, 8, sqrt(7.0)), (1, 2, 9, sqrt(7.0)),
]
const _TINY_CONV1_LS = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]

function _tiny_interaction!(st::_TinyStorages, k::Int)
    C, ns = _TINY_C, length(_TINY_ELEMENTS)
    attrs_ir = _terms(1, 0:3)
    hidden = _terms(C, 0:1)
    node_in = k == 0 ? _terms(C, 0:0) : hidden
    full = _terms(C, 0:3)
    out_ir = k == 0 ? hidden : _terms(C, 0:0)
    conv_out = k == 0 ? full : _terms(C, _TINY_CONV1_LS)
    conv_ins = if k == 0
        [(0, l, l, "uvu", sqrt(2l + 1.0), [C, 1]) for l in 0:3]
    else
        [(a, b, c, "uvu", pw, [C, 1]) for (a, b, c, pw) in _TINY_CONV1]
    end
    n_paths = length(conv_ins)
    lin_in = k == 0 ? full : conv_out
    lin_ins = k == 0 ? [(l, l) for l in 0:3] : [(i - 1, [0, 0, 1, 1, 1, 2, 2, 2, 3, 3][i]) for i in 1:10]
    skip_ins = [(l, 0, l, "uvw", sqrt(2l + 1.0) / sqrt(C * ns), [C, ns, C]) for l in 0:(k == 0 ? 3 : 0)]
    skip_in = k == 0 ? full : hidden
    skip_out = k == 0 ? full : _terms(C, 0:0)
    avg = _buffers(st, "avg_num_neighbors" => 10.0)
    return _tiny_module(
        ("mace.modules.blocks", k == 0 ? "RealAgnosticDensityInteractionBlock" : "RealAgnosticDensityResidualInteractionBlock");
        buffers = avg,
        modules = [
            "linear_up" => _tiny_linear!(st, node_in, node_in, [(i, i) for i in 0:(length(node_in) - 1)]),
            "conv_tp" => _tiny_tp!(
                st, ("e3nn.o3._tensor_product._tensor_product", "TensorProduct"), node_in, attrs_ir, conv_out, conv_ins; weighted = false
            ),
            "conv_tp_weights" => _tiny_radial_mlp!(st, [_TINY_NB, 4, 4, 4, C * n_paths]),
            "linear" => _tiny_linear!(st, lin_in, k == 0 ? full : _terms(C, 0:3), lin_ins),
            "skip_tp" => _tiny_tp!(
                st, ("e3nn.o3._tensor_product._sub", "FullyConnectedTensorProduct"), skip_in, _terms(ns, 0:0), skip_out, skip_ins; weighted = true
            ),
            "density_fn" => _tiny_module(
                ("e3nn.nn._fc", "FullyConnectedNet");
                modules = ["layer0" => _tiny_module(("e3nn.nn._fc", "_Layer"); params = ["weight" => _weight!(st, [_TINY_NB, 1])])]
            ),
        ],
        attrs = [
            "node_feats_irreps" => _pk_irreps(node_in), "edge_attrs_irreps" => _pk_irreps(attrs_ir),
            "edge_feats_irreps" => _pk_irreps(_terms(_TINY_NB, 0:0)), "target_irreps" => _pk_irreps(full),
            "hidden_irreps" => _pk_irreps(out_ir), "radial_MLP" => _pk_list(_pk_int(4), _pk_int(4), _pk_int(4)),
        ]
    )
end

function _tiny_pickle!(st::_TinyStorages)
    C, ns, nb = _TINY_C, length(_TINY_ELEMENTS), _TINY_NB
    scalars = _terms(C, 0:0)
    hidden = _terms(C, 0:1)
    embedding = _tiny_module(
        ("mace.modules.blocks", "LinearNodeEmbeddingBlock");
        modules = ["linear" => _tiny_linear!(st, _terms(ns, 0:0), scalars, [(0, 0)])]
    )
    radial = _tiny_module(
        ("mace.modules.blocks", "RadialEmbeddingBlock");
        modules = [
            "bessel_fn" => _tiny_module(
                ("mace.modules.radial", "BesselBasis");
                buffers = [
                    "bessel_weights" => _tiny_f64!(st, [nb], [n * pi / _TINY_R_MAX for n in 1:nb]),
                    _buffers(st, "r_max" => _TINY_R_MAX, "prefactor" => sqrt(2 / _TINY_R_MAX))...,
                ]
            ),
            "distance_transform" => _tiny_module(
                ("mace.modules.radial", "AgnesiTransform");
                buffers = [
                    _buffers(st, "q" => 0.9183, "p" => 4.5791, "a" => 1.0805)...,
                    "covalent_radii" => _tiny_f64!(st, [length(_TINY_COVALENT_RADII)], _TINY_COVALENT_RADII),
                ]
            ),
            "cutoff_fn" => _tiny_module(("mace.modules.radial", "PolynomialCutoff"); buffers = _buffers(st, "p" => 5.0, "r_max" => _TINY_R_MAX)),
        ]
    )
    pair_repulsion = _tiny_module(
        ("mace.modules.radial", "ZBLBasis");
        buffers = [
            _buffers(st, "r_max" => _TINY_R_MAX)...,
            "c" => _tiny_f64!(st, [4], [0.1818, 0.5099, 0.2802, 0.02817]),
            _buffers(st, "p" => 5.0)...,
            "covalent_radii" => _tiny_f64!(st, [length(_TINY_COVALENT_RADII)], _TINY_COVALENT_RADII),
            _buffers(st, "a_exp" => 0.3, "a_prefactor" => 0.4543)...,
        ]
    )
    e0 = _tiny_module(("mace.modules.blocks", "AtomicEnergiesBlock"); buffers = ["atomic_energies" => _tiny_f64!(st, [ns], [-1.11734008, -0.60025846])])
    scale_shift = _tiny_module(("mace.modules.blocks", "ScaleShiftBlock"); buffers = _buffers(st, "scale" => 0.7735790334431056, "shift" => 0.05))
    products = _modulelist(
        _tiny_product!(st, [0, 1], false, _tiny_linear!(st, hidden, hidden, [(0, 0), (1, 1)])),
        _tiny_product!(st, [0], true, _tiny_linear!(st, scalars, scalars, [(0, 0)])),
    )
    readout0 = _tiny_module(("mace.modules.blocks", "LinearReadoutBlock"); modules = ["linear" => _tiny_linear!(st, hidden, _terms(1, 0:0), [(0, 0)])])
    readout1 = _tiny_module(
        ("mace.modules.blocks", "NonLinearReadoutBlock");
        modules = [
            "linear_1" => _tiny_linear!(st, scalars, scalars, [(0, 0)]),
            "non_linearity" => _tiny_module(("e3nn.nn._activation", "Activation"); modules = ["acts" => _modulelist(_silu())]),
            "linear_2" => _tiny_linear!(st, scalars, _terms(1, 0:0), [(0, 0)]),
        ],
        attrs = ["hidden_irreps" => _pk_irreps(scalars)]
    )
    interactions = _modulelist(_tiny_interaction!(st, 0), _tiny_interaction!(st, 1))
    root = _tiny_module(
        ("mace.modules.models", "ScaleShiftMACE");
        buffers = [
            "atomic_numbers" => _tiny_i64!(st, [ns], _TINY_ELEMENTS),
            _buffers(st, "r_max" => _TINY_R_MAX)...,
            "num_interactions" => _tiny_i64!(st, Int[], [2]),
        ],
        modules = [
            "node_embedding" => embedding, "radial_embedding" => radial, "pair_repulsion_fn" => pair_repulsion,
            "atomic_energies_fn" => e0, "interactions" => interactions, "products" => products,
            "readouts" => _modulelist(readout0, readout1), "scale_shift" => scale_shift,
        ],
        attrs = ["heads" => _pk_list(_pk_str("default")), "pair_repulsion" => _pk_bool(true)]
    )
    return UInt8[0x80, 0x02, root..., UInt8('.')]
end

"""
    tiny_mace_checkpoint() -> Vector{UInt8}

A tiny ScaleShiftMACE of the supported architecture as the bytes of a torch checkpoint:
`C = 2` channels, elements `[1, 29]`, radial MLP `[4, 4, 4]`, 10 Bessel functions, one nonzero
entry in every U matrix. The parameter at position `k` of the Float64 storage is `0.1 sin(k)`.
"""
function tiny_mace_checkpoint()
    sizing = _TinyStorages(Float64[], Int64[], 0, 0)
    _tiny_pickle!(sizing)
    st = _TinyStorages(Float64[], Int64[], length(sizing.f64), length(sizing.i64))
    pkl = _tiny_pickle!(st)
    f64 = UInt8[]
    for x in st.f64
        _put!(f64, reinterpret(UInt64, x))
    end
    i64 = UInt8[]
    for x in st.i64
        _put!(i64, reinterpret(UInt64, x))
    end
    return _tiny_zip(
        Pair{String, Vector{UInt8}}[
            "archive/data.pkl" => pkl, "archive/byteorder" => Vector{UInt8}(codeunits("little")),
            "archive/data/0" => f64, "archive/data/1" => i64,
        ]
    )
end
