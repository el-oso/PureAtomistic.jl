@testmodule MDRunFixtures begin
    using PureMLIP, PureAtomistic, TOML

    # The tiny checkpoint (elements H and Cu) written to a file, as the binary reads it.
    function write_model(dir)
        path = joinpath(dir, "tiny.model")
        write(path, PureMLIP.tiny_mace_checkpoint())
        return path
    end

    base_config() = Dict{String, Any}(
        "steps" => 20, "dt" => 1.0, "seed" => 3, "temperature" => 300.0, "vacancy" => true, "output_every" => 5,
        "lattice" => Dict{String, Any}("type" => "fcc", "element" => "Cu", "a" => 3.61, "size" => [2, 2, 2]),
        "thermostat" => Dict{String, Any}("type" => "langevin", "gamma" => 0.01),
    )

    function write_config(dir, cfg::AbstractDict; name = "md.toml")
        path = joinpath(dir, name)
        open(io -> TOML.print(io, cfg), path, "w")
        return path
    end

    # Runs `run_md` on `cfg` and returns the trajectory path.
    function run_config(dir, model, cfg; name = "out.traj")
        out = joinpath(dir, name)
        PureAtomistic.run_md(model, write_config(dir, cfg), out)
        return out
    end
end

@testitem "run_md matches a hand-built simulation" setup = [MDRunFixtures] begin
    using PureAtomistic
    using PureMLIP, PureAtomistic.IO, PureAtomistic.Systems, Random
    mktempdir() do dir
        model = MDRunFixtures.write_model(dir)
        cfg = MDRunFixtures.base_config()
        out = MDRunFixtures.run_config(dir, model, cfg)

        # The same run assembled from the public pieces.
        sys = remove_atom(fcc_cell(3.61, (2, 2, 2), :Cu), 1)
        rng = Xoshiro(3)
        maxwell_boltzmann!(sys, 300.0, rng)
        m = PureMLIP.MACEModel{Float64}(PureMLIP.load_mace_checkpoint(model))
        pot = MACEPotential(m, sys.positions, sys.cell, sys.species)
        sim = Simulation(sys, pot, VelocityVerlet(1.0); thermostat = LangevinBAOAB(300.0, 0.01), rng)
        run!(sim, 20)

        r = TrajectoryReader(out)
        @test length(r) == 5 # frame 0, then steps 5, 10, 15, 20
        last = read_frame(r, 5)
        close(r)
        # Equality is exact: both paths run the same arithmetic and the trajectory stores Float32.
        @test last.positions == Float32.(sim.system.positions)
        @test last.velocities == Float32.(sim.system.velocities)
        @test last.time == 20.0
        @test last.energy == sim.potential.result.energy

        frames = open(read_extxyz, out * ".extxyz")
        @test length(frames) == 1
        @test frames[1].positions == sim.system.positions
        @test frames[1].forces == sim.potential.result.forces
        @test length(frames[1].species) == 31 && all(==("Cu"), frames[1].species)
    end
end

@testitem "run_md runs every thermostat and the barostat" setup = [MDRunFixtures] begin
    using PureAtomistic
    using PureAtomistic.IO
    mktempdir() do dir
        model = MDRunFixtures.write_model(dir)
        thermostats = (
            Dict{String, Any}("type" => "none"),
            Dict{String, Any}("type" => "langevin", "gamma" => 0.01),
            Dict{String, Any}("type" => "bussi", "tau" => 100),
        )
        barostat = Dict{String, Any}("type" => "berendsen", "pressure" => 0.0, "tau" => 1000.0, "compressibility" => 0.01)
        for th in thermostats, baro in (nothing, barostat)
            cfg = MDRunFixtures.base_config()
            cfg["steps"] = 3
            cfg["vacancy"] = false
            delete!(cfg, "output_every")
            cfg["thermostat"] = th
            isnothing(baro) || (cfg["barostat"] = baro)
            out = MDRunFixtures.run_config(dir, model, cfg)
            r = TrajectoryReader(out)
            @test length(r) == 4 # output_every defaults to 1
            close(r)
        end
        # A cold start has no velocities to draw.
        cfg = MDRunFixtures.base_config()
        cfg["temperature"] = 0
        cfg["thermostat"] = Dict{String, Any}("type" => "none")
        r = TrajectoryReader(MDRunFixtures.run_config(dir, model, cfg))
        @test all(iszero, read_frame(r, 1).velocities)
        close(r)
    end
end

@testitem "run_md rejects a bad config at the key" setup = [MDRunFixtures] begin
    using PureAtomistic
    using TOML
    mktempdir() do dir
        model = MDRunFixtures.write_model(dir)
        out = joinpath(dir, "out.traj")
        function failing(edit!)
            cfg = MDRunFixtures.base_config()
            edit!(cfg)
            return () -> PureAtomistic.run_md(model, MDRunFixtures.write_config(dir, cfg), out)
        end
        @test_throws "pureatomistic: missing config key \"steps\"" failing(c -> delete!(c, "steps"))()
        @test_throws "pureatomistic: missing config key \"lattice.a\"" failing(c -> delete!(c["lattice"], "a"))()
        @test_throws "pureatomistic: missing config key \"thermostat\"" failing(c -> delete!(c, "thermostat"))()
        @test_throws "pureatomistic: missing config key \"thermostat.gamma\"" failing(c -> delete!(c["thermostat"], "gamma"))()
        @test_throws "pureatomistic: unknown config key \"stpes\"" failing(c -> c["stpes"] = 3)()
        @test_throws "pureatomistic: unknown config key \"lattice.b\"" failing(c -> c["lattice"]["b"] = 3)()
        @test_throws "pureatomistic: unknown config key \"thermostat.tau\"" failing(c -> c["thermostat"]["tau"] = 3)()
        @test_throws "pureatomistic: unknown config key \"thermostat.gamma\"" failing(c -> c["thermostat"] = Dict{String, Any}("type" => "none", "gamma" => 1))()
        @test_throws "pureatomistic: unknown config key \"thermostat.gamma\"" failing(c -> c["thermostat"] = Dict{String, Any}("type" => "bussi", "tau" => 1, "gamma" => 1))()
        @test_throws "pureatomistic: config key \"steps\" must be an integer of at least 0, got -1" failing(c -> c["steps"] = -1)()
        @test_throws "pureatomistic: config key \"steps\" must be an integer, got 2.5" failing(c -> c["steps"] = 2.5)()
        @test_throws "pureatomistic: config key \"output_every\" must be an integer of at least 1, got 0" failing(c -> c["output_every"] = 0)()
        @test_throws "pureatomistic: config key \"dt\" must be positive, got 0.0" failing(c -> c["dt"] = 0.0)()
        @test_throws "pureatomistic: config key \"dt\" must be a finite number, got \"fast\"" failing(c -> c["dt"] = "fast")()
        @test_throws "pureatomistic: config key \"dt\" must be a finite number, got Inf" failing(c -> c["dt"] = Inf)()
        @test_throws "pureatomistic: config key \"temperature\" must be non-negative, got -1.0" failing(c -> c["temperature"] = -1.0)()
        @test_throws "pureatomistic: config key \"temperature\" must be positive with the bussi thermostat, got 0.0" failing(c -> (c["temperature"] = 0.0; c["thermostat"] = Dict{String, Any}("type" => "bussi", "tau" => 100.0)))()
        @test_throws "pureatomistic: config key \"seed\" must be an integer, got 1.5" failing(c -> c["seed"] = 1.5)()
        @test_throws "pureatomistic: config key \"vacancy\" must be true or false, got 1" failing(c -> c["vacancy"] = 1)()
        @test_throws "pureatomistic: config key \"lattice\" must be a table, got 3" failing(c -> c["lattice"] = 3)()
        @test_throws "pureatomistic: config key \"lattice.type\" must be \"fcc\", got \"bcc\"" failing(c -> c["lattice"]["type"] = "bcc")()
        @test_throws "pureatomistic: config key \"lattice.type\" must be a string, got 1" failing(c -> c["lattice"]["type"] = 1)()
        @test_throws "pureatomistic: config key \"lattice.element\" must be a built-in element symbol, got \"Xx\"" failing(c -> c["lattice"]["element"] = "Xx")()
        @test_throws "pureatomistic: config key \"lattice.a\" must be positive, got -3.6" failing(c -> c["lattice"]["a"] = -3.6)()
        @test_throws "pureatomistic: config key \"lattice.size\" must be an array of three positive integers, got an array of 2 elements" failing(c -> c["lattice"]["size"] = [2, 2])()
        @test_throws "pureatomistic: config key \"lattice.size\" must be an array of three positive integers, got an array of 3 elements" failing(c -> c["lattice"]["size"] = [2, 0, 2])()
        @test_throws "pureatomistic: config key \"thermostat.type\" must be \"none\", \"langevin\" or \"bussi\", got \"nose\"" failing(c -> c["thermostat"]["type"] = "nose")()
        @test_throws "pureatomistic: config key \"thermostat.gamma\" must be non-negative, got -0.1" failing(c -> c["thermostat"]["gamma"] = -0.1)()
        @test_throws "pureatomistic: config key \"thermostat.tau\" must be positive, got 0" failing(c -> c["thermostat"] = Dict{String, Any}("type" => "bussi", "tau" => 0))()
        baro(; kw...) = Dict{String, Any}("type" => "berendsen", "pressure" => 0.0, "tau" => 1000.0, "compressibility" => 0.01, (string(k) => v for (k, v) in kw)...)
        @test_throws "pureatomistic: config key \"barostat.type\" must be \"berendsen\", got \"mttk\"" failing(c -> c["barostat"] = baro(; type = "mttk"))()
        @test_throws "pureatomistic: config key \"barostat.tau\" must be positive, got -1.0" failing(c -> c["barostat"] = baro(; tau = -1.0))()
        @test_throws "pureatomistic: config key \"barostat.compressibility\" must be positive, got 0.0" failing(c -> c["barostat"] = baro(; compressibility = 0.0))()
        @test_throws "pureatomistic: unknown config key \"barostat.mass\"" failing(c -> c["barostat"] = baro(; mass = 1))()
        @test_throws "pureatomistic: cannot parse the config file" PureAtomistic.run_md(
            model, (p = joinpath(dir, "bad.toml"); write(p, "steps = = 3"); p), out
        )
    end
end

# Builds `md_run` with juliac --trim=safe (Julia 1.13) and runs 100 steps on the tiny checkpoint;
# the trajectory and the final frame must equal the untrimmed run bit for bit. Slow, so it runs
# only with PUREATOMISTIC_JULIAC_BUILD=1 (see test/runtests.jl).
@testitem "juliac --trim build of md_run" tags = [:juliac] setup = [MDRunFixtures] begin
    using PureAtomistic
    root = pkgdir(PureAtomistic)
    exe = joinpath(root, "juliac", "build", "bin", "md_run" * (Sys.iswindows() ? ".exe" : ""))
    # A TestItemRunner worker restricts JULIA_LOAD_PATH; the juliac subprocess needs the default
    # path (including @stdlib).
    cmd = addenv(
        `$(Base.julia_cmd()) --startup-file=no --project=$root $(joinpath(root, "juliac", "build.jl"))`,
        "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing
    )
    @test success(pipeline(cmd; stdout = stderr, stderr = stderr))
    @test isfile(exe)
    @test !isfile(joinpath(root, "LocalPreferences.toml"))

    dir = mktempdir()
    testenv = joinpath(root, "test")

    # Runs the binary under a timeout; returns (exit code, stdout text, stderr text). With
    # `strictmode_env` the process load path and active project name StrictMode, which makes a
    # StrictMode that was built with its banner on print it.
    function run_exe(args...; seconds = 600, strictmode_env = false)
        errf, outf = joinpath(dir, "stderr.txt"), joinpath(dir, "stdout.txt")
        env = strictmode_env ?
            ("JULIA_LOAD_PATH" => string("@:", testenv, ":@stdlib"), "JULIA_PROJECT" => testenv) :
            ("JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
        p = run(pipeline(ignorestatus(addenv(`$exe $args`, env...)); stdout = outf, stderr = errf); wait = false)
        if timedwait(() -> !process_running(p), seconds) !== :ok
            kill(p)
            error("md_run did not exit within $seconds s")
        end
        return p.exitcode, read(outf, String), read(errf, String)
    end

    model = MDRunFixtures.write_model(dir)
    base = MDRunFixtures.base_config()
    base["steps"] = 100
    base["output_every"] = 10

    # The binary equals the untrimmed run bit for bit, with nothing on stdout or stderr, for a
    # Langevin run under a StrictMode-naming environment and for Bussi with the barostat.
    bussi = deepcopy(base)
    bussi["thermostat"] = Dict{String, Any}("type" => "bussi", "tau" => 100.0)
    bussi["barostat"] = Dict{String, Any}("type" => "berendsen", "pressure" => 0.0, "tau" => 1000.0, "compressibility" => 0.01)
    for (name, cfg) in (("langevin", base), ("bussi", bussi))
        config = MDRunFixtures.write_config(dir, cfg; name = "$name.toml")
        ref = joinpath(dir, "$name-ref.traj")
        PureAtomistic.run_md(model, config, ref)
        out = joinpath(dir, "$name-bin.traj")
        code, stdout_text, stderr_text = run_exe(model, config, out; strictmode_env = name == "langevin")
        @test code == 0
        @test isempty(stdout_text)
        @test isempty(stderr_text)
        @test read(out) == read(ref)
        @test read(out * ".extxyz") == read(ref * ".extxyz")
        @test filesize(out) > 0
    end

    # Every failure exits 1 with a readable message on stderr and nothing on stdout.
    function fails(args...; message)
        code, stdout_text, stderr_text = run_exe(args...)
        @test code == 1
        @test isempty(stdout_text)
        @test occursin(message, stderr_text)
        @test !occursin("fatal:", stderr_text)
        return stderr_text
    end
    out = joinpath(dir, "fail.traj")
    good = MDRunFixtures.write_config(dir, base; name = "good.toml")
    fails(; message = "usage: md_run")
    bad = deepcopy(base)
    bad["steps"] = -1
    fails(model, MDRunFixtures.write_config(dir, bad; name = "bad.toml"), out; message = "pureatomistic: config key \"steps\" must be an integer of at least 0, got -1")
    fails(model, joinpath(dir, "absent.toml"), out; message = "absent.toml")
    fails(joinpath(dir, "absent.model"), good, out; message = "absent.model")
    gold = deepcopy(base)
    gold["lattice"]["element"] = "Au"
    fails(model, MDRunFixtures.write_config(dir, gold; name = "gold.toml"), out; message = "79")
end
