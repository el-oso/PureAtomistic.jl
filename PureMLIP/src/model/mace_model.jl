# A ScaleShiftMACE of the MACE-MP-0b3 / MACE-MPA-0 architecture, built from a checkpoint:
# two density-normalized interactions, correlation-3 products, a linear and a nonlinear readout,
# ZBL pair repulsion and an Agnesi-transformed Bessel radial basis. Channel, element, basis and
# radial-MLP sizes are read from the checkpoint; every other choice is checked against it.
# The forward pass follows mace (MIT), `mace/modules/models.py` and `mace/modules/blocks.py`.

const _INTERACTION_CLASSES = [
    "mace.modules.blocks.RealAgnosticDensityInteractionBlock",
    "mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock",
]
const _READOUT_CLASSES = ["mace.modules.blocks.LinearReadoutBlock", "mace.modules.blocks.NonLinearReadoutBlock"]
const _AGNESI = "mace.modules.radial.AgnesiTransform"
const _SCALE_SHIFT_MACE = "mace.modules.models.ScaleShiftMACE"
const _SILU = "torch.nn.functional.silu"

_unsupported(what, found, expected) = throw(ArgumentError(lazy"model: unsupported checkpoint: $what $found, expected $expected"))

_quoted(s::String) = string('"', s, '"')

function _f64(ck::MACECheckpoint, key::String)
    haskey(ck.state.f64, key) || throw(ArgumentError(lazy"model: unsupported checkpoint: tensor $key missing"))
    return ck.state.f64[key]
end

function _scalar(ck::MACECheckpoint, key::String)
    x = _f64(ck, key)
    length(x.data) == 1 || _unsupported(string("tensor ", key, " of size"), x.shape, "a scalar")
    return x.data[1]
end

function _spec(d::Dict{String, S}, key::String) where {S}
    haskey(d, key) || throw(ArgumentError(lazy"model: unsupported checkpoint: module $key missing"))
    return d[key]
end

# Degrees of the slots of `C` channels that hold e3nn irreps whose multiplicities are multiples
# of `C`, and the first slot of each irreps term.
function _slots(irreps::Vector{Checkpoint.IrrepMul}, C::Int, path::String)
    ls = Int[]
    first = Int[]
    for ir in irreps
        (ir.mul > 0 && ir.mul % C == 0) || _unsupported(string(path, " multiplicity"), ir.mul, string("a multiple of ", C))
        push!(first, length(ls) + 1)
        append!(ls, fill(ir.l, ir.mul ÷ C))
    end
    return ls, first
end

# The e3nn `Linear` at `path` over slots of `C_in` and `C_out` channels. An instruction whose
# input term holds `k` slots (MPA-0 stores `interactions.1.linear` over `256x0e+384x1o+...`)
# becomes `k` instructions with its path weight: a row-major `[k·C_in, C_out]` weight block is
# `k` consecutive `[C_in, C_out]` blocks.
function _linear_plan(ck::MACECheckpoint, path::String, C_in::Int, C_out::Int)
    spec = _spec(ck.constants.linear, path)
    in_ls, in_first = _slots(spec.irreps_in, C_in, path)
    out_ls, out_first = _slots(spec.irreps_out, C_out, path)
    ins = Tuple{Int, Int, Float64}[]
    for I in spec.instructions
        (I.i_in + 1 in eachindex(spec.irreps_in) && I.i_out + 1 in eachindex(spec.irreps_out)) ||
            _unsupported(string(path, " instruction"), (I.i_in, I.i_out), "indices within the irreps")
        mi = spec.irreps_in[I.i_in + 1].mul
        I.path_shape == [mi, C_out] && spec.irreps_out[I.i_out + 1].mul == C_out ||
            _unsupported(string(path, " instruction path shape"), I.path_shape, [mi, C_out])
        for j in 0:(mi ÷ C_in - 1)
            push!(ins, (in_first[I.i_in + 1] + j, out_first[I.i_out + 1], I.path_weight))
        end
    end
    return in_ls, out_ls, ins
end

function _linear(::Type{T}, ck::MACECheckpoint, path::String, C_in::Int, C_out::Int) where {T}
    in_ls, out_ls, ins = _linear_plan(ck, path, C_in, C_out)
    return E3nnLinear{T}(_f64(ck, path * ".weight").data, in_ls, out_ls, ins, C_in, C_out)
end

"""
    symmetric_contraction_from_state(st, prefix, T) -> SymmetricContraction{T}

The contractions `prefix.symmetric_contractions.contractions.0, 1, ...` of a MACE state dict,
in order, pairing `weights_max` with `U_matrix_3`, `weights.0` with `U_matrix_2` and `weights.1`
with `U_matrix_1`. The order `l` of each contraction is read from the shape of `U_matrix_1`
(two dimensions for `l = 0`, a leading `2l + 1` otherwise).
"""
function symmetric_contraction_from_state(st::Dict{String, CheckpointTensor{Float64}}, prefix::String, ::Type{T}) where {T}
    contractions = Contraction{T}[]
    j = 0
    while true
        p = string(prefix, ".symmetric_contractions.contractions.", j) * "."
        haskey(st, p * "U_matrix_1") || break
        u1, u2, u3 = _tensor(st, p * "U_matrix_1"), _tensor(st, p * "U_matrix_2"), _tensor(st, p * "U_matrix_3")
        Wmax = torch_array(_tensor(st, p * "weights_max"), Val(3))
        W0 = torch_array(_tensor(st, p * "weights.0"), Val(3))
        W1 = torch_array(_tensor(st, p * "weights.1"), Val(3))
        ct = if length(u1.shape) == 3
            Contraction{T}((u1.shape[1] - 1) ÷ 2, torch_array(u1, Val(3)), torch_array(u2, Val(4)), torch_array(u3, Val(5)), Wmax, W0, W1)
        else
            Contraction{T}(0, torch_array(u1, Val(2)), torch_array(u2, Val(3)), torch_array(u3, Val(4)), Wmax, W0, W1)
        end
        push!(contractions, ct)
        j += 1
    end
    j > 0 || throw(ArgumentError("model: missing $prefix.symmetric_contractions.contractions.0.U_matrix_1"))
    return SymmetricContraction{T}(contractions)
end

function _tensor(st::Dict{String, CheckpointTensor{Float64}}, key::String)
    haskey(st, key) || throw(ArgumentError("model: missing $key"))
    return st[key]
end

# Degrees of the irreps terms, each of multiplicity `mul`.
function _degrees(irreps::Vector{Checkpoint.IrrepMul}, mul::Int, path::String)
    for ir in irreps
        ir.mul == mul || _unsupported(string(path, " multiplicity"), ir.mul, mul)
    end
    return [ir.l for ir in irreps]
end

function _conv_tp(::Type{T}, ck::MACECheckpoint, path::String, C::Int) where {T}
    spec = _spec(ck.constants.tensor_product, path)
    ins = Tuple{Int, Int, Int, Float64}[]
    for I in spec.instructions
        I.connection_mode == "uvu" || _unsupported(string(path, " connection mode"), _quoted(I.connection_mode), _quoted("uvu"))
        push!(ins, (I.i_in1 + 1, I.i_in2 + 1, I.i_out + 1, I.path_weight))
    end
    return ChannelwiseTP{T}(
        _degrees(spec.irreps_in1, C, path), _degrees(spec.irreps_in2, 1, path), _degrees(spec.irreps_out, C, path), ins
    )
end

# The `skip_tp` of an interaction: a species-indexed linear map whose output slots are the first
# input slots, so it acts on the leading components of the block input.
function _skip_tp(::Type{T}, ck::MACECheckpoint, path::String, C::Int, n_species::Int) where {T}
    spec = _spec(ck.constants.tensor_product, path)
    ls = _degrees(spec.irreps_out, C, path)
    in_ls = _degrees(spec.irreps_in1, C, path)
    _degrees(spec.irreps_in2, n_species, path) == [0] || _unsupported(string(path, " second input"), spec.irreps_in2, string(n_species, "x0e"))
    (length(in_ls) >= length(ls) && in_ls[eachindex(ls)] == ls) ||
        _unsupported(string(path, " input degrees starting"), in_ls, ls)
    ins = Tuple{Int, Int, Float64}[]
    for I in spec.instructions
        (I.connection_mode == "uvw" && I.i_in2 == 0 && I.path_shape == [C, n_species, C]) ||
            _unsupported(string(path, " instruction"), (I.connection_mode, I.i_in2, I.path_shape), ("uvw", 0, [C, n_species, C]))
        push!(ins, (I.i_in1 + 1, I.i_out + 1, I.path_weight))
    end
    return SpeciesLinear{T}(_f64(ck, path * ".weight").data, ls, ins, C, n_species, C)
end

function _radial_mlp(::Type{T}, ck::MACECheckpoint, k::Int, cst::Float64) where {T}
    p = string("interactions.", k, ".conv_tp_weights.layer")
    layers = ntuple(j -> torch_array(_f64(ck, string(p, j - 1, ".weight")), Val(2)), Val(4))
    return RadialMLP{T}(layers, cst)
end

function _validate(ck::MACECheckpoint)
    h, k = ck.hyperparameters, ck.constants
    h.model_class == _SCALE_SHIFT_MACE || _unsupported("model class", _quoted(h.model_class), _quoted(_SCALE_SHIFT_MACE))
    classes = [i.class for i in h.interactions]
    classes == _INTERACTION_CLASSES || _unsupported("interaction classes", classes, _INTERACTION_CLASSES)
    h.correlations == [3, 3] || _unsupported("correlations", h.correlations, [3, 3])
    readouts = [r.class for r in h.readouts]
    readouts == _READOUT_CLASSES || _unsupported("readout classes", readouts, _READOUT_CLASSES)
    length(h.heads) == 1 || _unsupported("heads", h.heads, "one head")
    hidden = h.interactions[1].hidden_irreps
    C = isempty(hidden) ? 0 : hidden[1].mul
    hidden == [Checkpoint.IrrepMul(C, 0, 1), Checkpoint.IrrepMul(C, 1, -1)] && C > 0 ||
        _unsupported("hidden irreps", hidden, "Cx0e+Cx1o")
    attrs = [Checkpoint.IrrepMul(1, l, iseven(l) ? 1 : -1) for l in 0:3]
    for i in h.interactions
        i.edge_attrs_irreps == attrs || _unsupported("edge attributes", i.edge_attrs_irreps, "1x0e+1x1o+1x2e+1x3o")
    end
    k.pair_repulsion || _unsupported("pair_repulsion", false, true)
    k.distance_transform == _AGNESI || _unsupported("distance transform", _quoted(k.distance_transform), _quoted(_AGNESI))
    isempty(k.normalize2mom) && _unsupported("normalize2mom modules", 0, "at least one")
    cst = first(values(k.normalize2mom)).cst
    for n in values(k.normalize2mom)
        n.f == _SILU || _unsupported("normalize2mom function", _quoted(n.f), _quoted(_SILU))
        n.cst == cst || _unsupported("normalize2mom constants", (n.cst, cst), "one value")
    end
    k.product_use_sc == [false, true] || _unsupported("product use_sc", k.product_use_sc, Bool[false, true])
    return C, cst
end

"""
    MACEModel{T}(ck::MACECheckpoint)
    MACEModel(ck::MACECheckpoint) = MACEModel{Float64}(ck)

A ScaleShiftMACE model of the MACE-MP-0b3 / MACE-MPA-0 architecture with parameters of type
`T`, read from a checkpoint; any other architecture throws `model: unsupported checkpoint`.
Public fields: `z_table` (the atomic numbers of the elements, in species order), `r_max` (the
cutoff in Å), `e0` (the reference energy of each element, in eV), `scale` and `shift` (the
interaction energy of an atom is `scale · E + shift`).
"""
struct MACEModel{T <: AbstractFloat}
    z_table::Vector{Int}
    r_max::T
    e0::Vector{T}
    scale::T
    shift::T
    species::Vector{Int}                  # species[Z + 1]: position of Z in z_table, 0 when absent
    C::Int
    embedding::Matrix{T}                  # [C, n_species]: path weight · node_embedding weight
    radial::RadialEmbedding{T}
    zbl::ZBLParams{T}
    up::NTuple{2, E3nnLinear{T}}
    mlp::NTuple{2, RadialMLP{T}}
    density::NTuple{2, Vector{T}}
    conv::NTuple{2, ChannelwiseTP{T}}
    linear::NTuple{2, E3nnLinear{T}}
    skip::NTuple{2, SpeciesLinear{T}}
    product::NTuple{2, SymmetricContraction{T}}
    product_linear::NTuple{2, E3nnLinear{T}}
    readout0::E3nnLinear{T}
    readout1::NTuple{2, E3nnLinear{T}}
    readout_cst::T
    function MACEModel{T}(ck::MACECheckpoint) where {T <: AbstractFloat}
        C, cst = _validate(ck)
        h = ck.hyperparameters
        z_table = copy(h.atomic_numbers)
        ns = length(z_table)
        species = zeros(Int, maximum(z_table; init = 0) + 1)
        for (s, z) in pairs(z_table)
            z >= 0 || _unsupported("atomic number", z, "a non-negative integer")
            species[z + 1] = s
        end
        e0 = _f64(ck, "atomic_energies_fn.atomic_energies").data
        length(e0) == ns || _unsupported("atomic energies of length", length(e0), ns)

        emb_spec = _spec(ck.constants.linear, "node_embedding.linear")
        emb_shapes = [I.path_shape for I in emb_spec.instructions]
        emb_shapes == [[ns, C]] || _unsupported("node_embedding.linear path shapes", emb_shapes, [[ns, C]])
        W_emb = torch_array(_f64(ck, "node_embedding.linear.weight"), Val(1))
        length(W_emb) == ns * C || _unsupported("node_embedding.linear.weight of length", length(W_emb), ns * C)
        pw = emb_spec.instructions[1].path_weight
        embedding = T[pw * W_emb[(s - 1) * C + w] for w in 1:C, s in 1:ns]

        p = _scalar(ck, "radial_embedding.cutoff_fn.p")
        isinteger(p) || _unsupported("cutoff_fn.p", p, "an integer")
        r_max = _scalar(ck, "radial_embedding.bessel_fn.r_max")
        r_max == h.r_max == _scalar(ck, "radial_embedding.cutoff_fn.r_max") || _unsupported("radial r_max", r_max, h.r_max)
        bessel = _f64(ck, "radial_embedding.bessel_fn.bessel_weights").data
        radial = RadialEmbedding{T}(
            bessel, _scalar(ck, "radial_embedding.bessel_fn.prefactor"), r_max, Int(p),
            _scalar(ck, "radial_embedding.distance_transform.a"), _scalar(ck, "radial_embedding.distance_transform.p"),
            _scalar(ck, "radial_embedding.distance_transform.q"), _f64(ck, "radial_embedding.distance_transform.covalent_radii").data
        )
        c = _f64(ck, "pair_repulsion_fn.c").data
        length(c) == 4 || _unsupported("pair_repulsion_fn.c of length", length(c), 4)
        zbl = ZBLParams{T}(
            (c[1], c[2], c[3], c[4]), _scalar(ck, "pair_repulsion_fn.p"), _scalar(ck, "pair_repulsion_fn.a_exp"),
            _scalar(ck, "pair_repulsion_fn.a_prefactor"), _f64(ck, "pair_repulsion_fn.covalent_radii").data
        )

        up = (_linear(T, ck, "interactions.0.linear_up", C, C), _linear(T, ck, "interactions.1.linear_up", C, C))
        mlp = (_radial_mlp(T, ck, 0, cst), _radial_mlp(T, ck, 1, cst))
        density = ntuple(j -> T.(_f64(ck, string("interactions.", j - 1, ".density_fn.layer0.weight")).data), Val(2))
        conv = (_conv_tp(T, ck, "interactions.0.conv_tp", C), _conv_tp(T, ck, "interactions.1.conv_tp", C))
        # interactions.1.linear: one instruction per conv_tp output slot, in slot order.
        tp_ls = _degrees(_spec(ck.constants.tensor_product, "interactions.1.conv_tp").irreps_out, C, "interactions.1.conv_tp")
        lin1_ls, _, lin1_ins = _linear_plan(ck, "interactions.1.linear", C, C)
        lin1_in = [i for (i, _, _) in lin1_ins]
        (lin1_ls == tp_ls && lin1_in == collect(eachindex(tp_ls))) ||
            _unsupported("interactions.1.linear input slots", lin1_in, collect(eachindex(tp_ls)))
        linear = (_linear(T, ck, "interactions.0.linear", C, C), _linear(T, ck, "interactions.1.linear", C, C))
        skip = (_skip_tp(T, ck, "interactions.0.skip_tp", C, ns), _skip_tp(T, ck, "interactions.1.skip_tp", C, ns))
        product = (
            symmetric_contraction_from_state(ck.state.f64, "products.0", T),
            symmetric_contraction_from_state(ck.state.f64, "products.1", T),
        )
        product_linear = (_linear(T, ck, "products.0.linear", C, C), _linear(T, ck, "products.1.linear", C, C))
        readout0 = _linear(T, ck, "readouts.0.linear", C, 1)
        H = h.readouts[2].hidden_irreps
        (length(H) == 1 && H[1].l == 0) || _unsupported("readouts.1 hidden irreps", H, "Hx0e")
        readout1 = (_linear(T, ck, "readouts.1.linear_1", C, H[1].mul), _linear(T, ck, "readouts.1.linear_2", H[1].mul, 1))

        # The workspace holds one radial-MLP buffer set for both interactions.
        size(mlp[1].W[1], 2) == size(mlp[2].W[1], 2) ||
            _unsupported("interactions.1.conv_tp_weights hidden width", size(mlp[2].W[1], 2), size(mlp[1].W[1], 2))

        # The feature shapes that connect the blocks.
        nb = length(bessel)
        for j in 1:2
            size(mlp[j].W[1], 1) == nb || _unsupported(string("interactions.", j - 1, ".conv_tp_weights inputs"), size(mlp[j].W[1], 1), nb)
            size(mlp[j].W[4], 2) == C * length(conv[j].paths) ||
                _unsupported(string("interactions.", j - 1, ".conv_tp_weights outputs"), size(mlp[j].W[4], 2), C * length(conv[j].paths))
            length(density[j]) == nb || _unsupported(string("interactions.", j - 1, ".density_fn inputs"), length(density[j]), nb)
            conv[j].K_in2 == 16 || _unsupported(string("interactions.", j - 1, ".conv_tp attributes"), conv[j].K_in2, 16)
        end
        conv[1].K_in1 == up[1].K_out == 1 || _unsupported("interactions.0 input components", conv[1].K_in1, 1)
        conv[2].K_in1 == up[2].K_out == 4 || _unsupported("interactions.1 input components", conv[2].K_in1, 4)
        conv[1].K_out == linear[1].K_in && conv[2].K_out == linear[2].K_in ||
            _unsupported("interaction linear inputs", (linear[1].K_in, linear[2].K_in), (conv[1].K_out, conv[2].K_out))
        linear[1].K_out == linear[2].K_out == skip[1].K == 16 || _unsupported("interaction outputs", linear[1].K_out, 16)
        (n_outputs(product[1]) == 4 && n_outputs(product[2]) == 1) ||
            _unsupported("product outputs", (n_outputs(product[1]), n_outputs(product[2])), (4, 1))
        product_linear[1].K_in == 4 && product_linear[2].K_in == 1 && skip[2].K == 1 && readout0.K_in == 4 ||
            _unsupported("product linear inputs", (product_linear[1].K_in, product_linear[2].K_in), (4, 1))

        return new{T}(
            z_table, T(r_max), T.(e0), T(_scalar(ck, "scale_shift.scale")), T(_scalar(ck, "scale_shift.shift")),
            species, C, embedding, radial, zbl, up, mlp, density, conv, linear, skip, product, product_linear,
            readout0, readout1, T(cst),
        )
    end
end

MACEModel(ck::MACECheckpoint) = MACEModel{Float64}(ck)
