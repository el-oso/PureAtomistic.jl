# A molecular-dynamics run described by a TOML file; see `run_md`. The reader accepts exactly the
# documented keys and checks each value's type and range, so a wrong file fails at the key.

struct MDConfig
    element::String
    a::Float64
    size::NTuple{3, Int}
    vacancy::Bool
    temperature::Float64
    thermostat::Union{Nothing, LangevinBAOAB, BussiThermostat}
    barostat::Union{Nothing, BerendsenBarostat}
    steps::Int
    dt::Float64
    seed::Int
    every::Int
end

function _check_keys(tbl::Dict{String, Any}, where_::String, allowed)
    for k in keys(tbl)
        k in allowed ||
            throw(ArgumentError(string("pureatomistic: unknown config key \"", _key(where_, k), "\"")))
    end
    return nothing
end

_key(where_::String, k::String) = isempty(where_) ? k : string(where_, ".", k)

function _require(tbl::Dict{String, Any}, where_::String, k::String)
    haskey(tbl, k) || throw(ArgumentError(string("pureatomistic: missing config key \"", _key(where_, k), "\"")))
    return tbl[k]
end

# The offending value as written in the file, for the scalars and arrays a config holds.
_show(v) = v isa String ? string('"', v, '"') : v isa Union{Int, Float64, Bool} ? string(v) : "a value of another type"
_show_value(v) = v isa Vector ? string("an array of ", length(v), " elements") : _show(v)

function _wrong(where_::String, k::String, what, v)
    throw(ArgumentError(string("pureatomistic: config key \"", _key(where_, k), "\" must be ", what, ", got ", _show_value(v))))
end

function _table(tbl::Dict{String, Any}, where_::String, k::String)::Dict{String, Any}
    v = _require(tbl, where_, k)
    v isa Dict{String, Any} || _wrong(where_, k, "a table", v)
    return v
end

function _string(tbl::Dict{String, Any}, where_::String, k::String)::String
    v = _require(tbl, where_, k)
    v isa String || _wrong(where_, k, "a string", v)
    return v
end

function _bool(tbl::Dict{String, Any}, where_::String, k::String)::Bool
    v = _require(tbl, where_, k)
    v isa Bool || _wrong(where_, k, "true or false", v)
    return v
end

function _int(tbl::Dict{String, Any}, where_::String, k::String; min::Int)::Int
    v = _require(tbl, where_, k)
    v isa Int || _wrong(where_, k, string("an integer of at least ", min), v)
    v >= min || _wrong(where_, k, string("an integer of at least ", min), v)
    return v
end

# A finite number; an integer is accepted for a float key.
function _float(tbl::Dict{String, Any}, where_::String, k::String; positive::Bool = false, nonnegative::Bool = false)::Float64
    v = _require(tbl, where_, k)
    ((v isa Float64 || v isa Int) && isfinite(v)) || _wrong(where_, k, "a finite number", v)
    x = Float64(v)
    positive && !(x > 0) && _wrong(where_, k, "positive", v)
    nonnegative && !(x >= 0) && _wrong(where_, k, "non-negative", v)
    return x
end

function _size(tbl::Dict{String, Any}, where_::String, k::String)::NTuple{3, Int}
    v = _require(tbl, where_, k)
    v isa Vector{Int} || _wrong(where_, k, "an array of three positive integers", v)
    (length(v) == 3 && all(>=(1), v)) || _wrong(where_, k, "an array of three positive integers", v)
    return (v[1], v[2], v[3])
end

function _read_thermostat(tbl::Dict{String, Any}, temperature::Float64)
    th = _table(tbl, "", "thermostat")
    kind = _string(th, "thermostat", "type")
    if kind == "none"
        _check_keys(th, "thermostat", ("type",))
        return nothing
    elseif kind == "langevin"
        _check_keys(th, "thermostat", ("type", "gamma"))
        return LangevinBAOAB(temperature, _float(th, "thermostat", "gamma"; nonnegative = true))
    elseif kind == "bussi"
        _check_keys(th, "thermostat", ("type", "tau"))
        return BussiThermostat(temperature, _float(th, "thermostat", "tau"; positive = true))
    end
    return _wrong("thermostat", "type", "\"none\", \"langevin\" or \"bussi\"", kind)
end

function _read_barostat(tbl::Dict{String, Any})
    haskey(tbl, "barostat") || return nothing
    b = _table(tbl, "", "barostat")
    kind = _string(b, "barostat", "type")
    kind == "berendsen" || _wrong("barostat", "type", "\"berendsen\"", kind)
    _check_keys(b, "barostat", ("type", "pressure", "tau", "compressibility"))
    return BerendsenBarostat(
        _float(b, "barostat", "pressure"), _float(b, "barostat", "tau"; positive = true),
        _float(b, "barostat", "compressibility"; positive = true)
    )
end

function _read_config(tbl::Dict{String, Any})
    _check_keys(
        tbl, "", ("lattice", "vacancy", "temperature", "thermostat", "barostat", "steps", "dt", "seed", "output_every")
    )
    lat = _table(tbl, "", "lattice")
    _check_keys(lat, "lattice", ("type", "element", "a", "size"))
    kind = _string(lat, "lattice", "type")
    kind == "fcc" || _wrong("lattice", "type", "\"fcc\"", kind)
    element = _string(lat, "lattice", "element")
    try
        Systems.atomic_number(Symbol(element))
    catch err
        err isa ArgumentError || rethrow()
        _wrong("lattice", "element", "a built-in element symbol", element)
    end
    temperature = _float(tbl, "", "temperature"; nonnegative = true)
    steps = _int(tbl, "", "steps"; min = 0)
    every = haskey(tbl, "output_every") ? _int(tbl, "", "output_every"; min = 1) : 1
    return MDConfig(
        element, _float(lat, "lattice", "a"; positive = true), _size(lat, "lattice", "size"),
        _bool(tbl, "", "vacancy"), temperature, _read_thermostat(tbl, temperature), _read_barostat(tbl),
        steps, _float(tbl, "", "dt"; positive = true), _int(tbl, "", "seed"; min = typemin(Int)), every,
    )
end

function _read_config_file(path::AbstractString)
    parsed = TOML.tryparsefile(path)
    parsed isa TOML.ParserError &&
        throw(ArgumentError(string("pureatomistic: cannot parse the config file ", path, " at line ", parsed.line, ", column ", parsed.column)))
    return _read_config(parsed)
end

# Frame 0 and every `every`-th step of the run go to the trajectory.
function _simulate(cfg::MDConfig, system, potential, thermostat, barostat, rng, traj::IO.TrajectoryWriter)
    dt = cfg.dt
    sim = Simulation(system, potential, VelocityVerlet(dt); thermostat, barostat, rng)
    emit = s -> IO.write_frame(
        traj, s.step_count * dt, s.system.positions;
        velocities = s.system.velocities, cell = s.system.cell, energy = s.potential.result.energy
    )
    emit(sim)
    run!(sim, cfg.steps; every = cfg.every, observer = emit)
    return sim
end

function _simulate(cfg::MDConfig, system, potential, rng, traj::IO.TrajectoryWriter)
    th = cfg.thermostat
    if th isa LangevinBAOAB
        return _simulate_b(cfg, system, potential, th, rng, traj)
    elseif th isa BussiThermostat
        return _simulate_b(cfg, system, potential, th, rng, traj)
    end
    return _simulate_b(cfg, system, potential, nothing, rng, traj)
end

function _simulate_b(cfg::MDConfig, system, potential, th, rng, traj::IO.TrajectoryWriter)
    ba = cfg.barostat
    ba isa BerendsenBarostat && return _simulate(cfg, system, potential, th, ba, rng, traj)
    return _simulate(cfg, system, potential, th, nothing, rng, traj)
end

"""
    run_md(model_path, config_path, traj_path) -> nothing

Read the MACE checkpoint `model_path` and the TOML file `config_path`, build the system,
integrate it, and write the binary trajectory `traj_path` ([`IO.TrajectoryWriter`](@ref), positions
and velocities as `Float32`) and the final frame, with forces, to `traj_path * ".extxyz"`. The
arithmetic is `Float64`; a run is deterministic given the file and `seed`. A missing, unknown or
invalid key throws an `ArgumentError` naming the key.

The config file is flat except for the tables below; every key is required unless marked optional.

```toml
steps = 100              # integer >= 0
dt = 1.0                 # fs
seed = 1                 # integer, seeds the velocities and the thermostat
temperature = 300.0      # K, initial velocities and thermostat target
vacancy = false          # true removes atom 1
output_every = 10        # optional, default 1: trajectory frame every this many steps

[lattice]
type = "fcc"
element = "Cu"
a = 3.61                 # Å
size = [2, 2, 2]         # conventional cells

[thermostat]
type = "langevin"        # "none", "langevin" (gamma, 1/fs) or "bussi" (tau, fs)
gamma = 0.01

[barostat]               # optional
type = "berendsen"
pressure = 0.0           # GPa
tau = 1000.0             # fs
compressibility = 0.01   # 1/GPa
```
"""
function run_md(model_path::AbstractString, config_path::AbstractString, traj_path::AbstractString)::Nothing
    cfg = _read_config_file(config_path)
    model = PureMLIP.MACEModel{Float64}(PureMLIP.load_mace_checkpoint(model_path))
    system = Systems.fcc_cell(cfg.a, cfg.size, Symbol(cfg.element))
    cfg.vacancy && (system = Systems.remove_atom(system, 1))
    rng = Random.Xoshiro(cfg.seed)
    cfg.temperature > 0 && Systems.maxwell_boltzmann!(system, cfg.temperature, rng)
    potential = MACEPotential(model, system.positions, system.cell, system.species)
    traj = IO.TrajectoryWriter(traj_path, natoms(system), fill(cfg.element, natoms(system)))
    sim = try
        _simulate(cfg, system, potential, rng, traj)
    finally
        close(traj)
    end
    frame = IO.Frame(
        fill(cfg.element, natoms(system)), sim.system.positions;
        forces = sim.potential.result.forces, velocities = sim.system.velocities, cell = sim.system.cell,
        energy = sim.potential.result.energy, time = sim.step_count * cfg.dt
    )
    open(traj_path * ".extxyz", "w") do io
        IO.write_extxyz(io, frame)
    end
    return nothing
end
