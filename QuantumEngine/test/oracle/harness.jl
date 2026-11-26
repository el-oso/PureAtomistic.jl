# The parity harness: locates and runs the oracle binary, reads its outputs, and reads the stored
# references (`references.toml`) and tolerances (`tolerances.toml`). `capture.jl` loads this
# module outside the test runner to write the captured files.
@testmodule OracleHarness begin
    using QuantumEngine
    using SHA: sha256
    using TOML

    const ORACLE_DIR = joinpath(pkgdir(QuantumEngine), "test", "oracle")
    const REFERENCES = joinpath(ORACLE_DIR, "references.toml")
    const TOLERANCES = joinpath(ORACLE_DIR, "tolerances.toml")
    const CAPTURED = joinpath(ORACLE_DIR, "captured")

    # The vendored input files whose echoed settings are cross-checked, each with whether it parses
    # only with `lenient = true`.
    const FIXTURE_FILES = [
        "examples/h2.inp" => false,
        "examples/h2o_631g_sp.inp" => false,
        "examples/o2_triplet.inp" => true,
        "examples/h2o_tddft.inp" => false,
        "tests/h2o_dimer_bsse.inp" => false,
        "tests/o2_hubbard.in" => true,
    ]

    """
        captured(file) -> String

    Path of the captured oracle output `file`, relative to `test/oracle/captured`.
    """
    captured(file::AbstractString) = joinpath(CAPTURED, file)

    """
        fixture_stdout(relpath) -> String

    Path of the captured standard output of the vendored input file `relpath`.
    """
    fixture_stdout(relpath::AbstractString) =
        captured(joinpath("fixtures", splitext(basename(relpath))[1] * ".stdout"))

    """
        file_sha256(path) -> String

    The SHA-256 of the file at `path`, as 64 lowercase hexadecimal digits.
    """
    file_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

    # ---------------------------------------------------------------- locating the oracle

    """
        oracle_binary() -> Union{Nothing,String}

    The oracle binary named by `QUANTUM_ENGINE_ORACLE`, or `nothing` when it is unset or empty.
    """
    function oracle_binary()
        b = get(ENV, "QUANTUM_ENGINE_ORACLE", "")
        return isempty(b) ? nothing : b
    end

    """
        oracle_basis_dir() -> String

    The oracle's `basis_sets` directory, named by `QUANTUM_ENGINE_ORACLE_BASIS`.
    """
    function oracle_basis_dir()
        d = get(ENV, "QUANTUM_ENGINE_ORACLE_BASIS", "")
        if isempty(d)
            oracle_binary() === nothing &&
                error("oracle: QUANTUM_ENGINE_ORACLE_BASIS (the oracle's basis_sets directory) is not set")
            error("oracle: QUANTUM_ENGINE_ORACLE is set but QUANTUM_ENGINE_ORACLE_BASIS (the oracle's basis_sets directory) is not")
        end
        isdir(d) || error("oracle: QUANTUM_ENGINE_ORACLE_BASIS is not a directory")
        return d
    end

    # ---------------------------------------------------------------- running the oracle

    """
        run_oracle_file(filename, text; dir = mktempdir(), command = "run")
            -> NamedTuple{(:log, :stdout, :exitcode), Tuple{String,String,Int}}

    Write `text` to `dir/filename`, link `dir/basis_sets` to [`oracle_basis_dir`](@ref), and run
    `quantum_engine <command> <filename>` with `dir` as the working directory: the oracle resolves
    basis files relative to its working directory. Returns the log, the captured standard output
    and the exit code. Files the run writes besides these stay in `dir`.

    `command` is one of [`ORACLE_COMMANDS`](@ref):
    - `run` writes its log next to the input (`engine/mod.rs:35-39`: the input path with the
      extension `log`). The post-SCF modules print with `println!`, so their lines appear only in
      the standard output.
    - `aimext` reads the `.inp` file, computes the SCF density when no `<stem>.chk` lies next to it
      (and then saves that checkpoint), and writes the critical-point report `<stem>.ext` into the
      working directory (`main.rs:229-258`, `analysis/aim.rs:722-821`, `analysis/aim.rs:334-336`).
      Its writer has no log file (`main.rs:230`), so everything it prints is in the standard
      output, and `log` is the empty string.
    """
    function run_oracle_file(
            filename::AbstractString, text::AbstractString;
            dir::AbstractString = mktempdir(), command::AbstractString = "run",
        )
        command in ORACLE_COMMANDS || error("oracle: unknown command `$command`; expected one of $(join(ORACLE_COMMANDS, ", "))")
        bin = oracle_binary()
        bin === nothing && error("oracle: QUANTUM_ENGINE_ORACLE (the oracle binary) is not set")
        basis = oracle_basis_dir()
        write(joinpath(dir, filename), text)
        symlink(basis, joinpath(dir, "basis_sets"))
        out = IOBuffer()
        cmd = Cmd(`$bin $command $filename`; dir = String(dir))
        p = run(pipeline(ignorestatus(cmd); stdin = devnull, stdout = out, stderr = stderr))
        log = ""
        if writes_log(command)
            logfile = joinpath(dir, splitext(filename)[1] * ".log")
            isfile(logfile) || error("oracle: the run of $filename wrote no log (exit code $(p.exitcode))")
            log = read(logfile, String)
        end
        return (log = log, stdout = String(take!(out)), exitcode = Int(p.exitcode))
    end

    """
        ORACLE_COMMANDS

    The oracle subcommands the harness runs: `run`, which carries out the input's calculation, and
    `aimext`, the critical-point analysis of the input's SCF density.
    """
    const ORACLE_COMMANDS = ("run", "aimext")

    """
        writes_log(command) -> Bool

    Whether the oracle subcommand `command` writes a log file: `run` does, `aimext` does not.
    """
    writes_log(command::AbstractString) = command == "run"

    """
        case_command(case) -> String

    The oracle subcommand that runs `case`, its `command` in `references.toml`.
    """
    function case_command(case::AbstractString)
        c = case_table(case)["command"]
        c in ORACLE_COMMANDS || error("oracle: case $case names the unknown command `$c`")
        return c
    end

    """
        run_oracle(case; dir = mktempdir())
            -> NamedTuple{(:log, :stdout, :exitcode), Tuple{String,String,Int}}

    Run the oracle on the input of `case` in `references.toml`, written as `<case>.inp` into
    `dir`, with the case's [`case_command`](@ref). See [`run_oracle_file`](@ref).
    """
    run_oracle(case::String; dir::AbstractString = mktempdir()) =
        run_oracle_file(case * ".inp", case_input(case); dir, command = case_command(case))

    # The lines that change from run to run: elapsed times (`analysis/qtaim.rs:595`,
    # `analysis/aim.rs:340, 502, 699`, `analysis/iqa.rs:290`, `pw/scf.rs:97, 335`,
    # `dynamics/bomd.rs:301`).
    const TIMING = r"(elapsed:|completed in|\(took|ready in|Total MD Time:)\s*[0-9]+(\.[0-9]+)?\s*(s|ms|µs|ns)\b"

    """
        scrub(text, name, dir) -> String

    `text` with the lines that depend on the machine or the run removed or rewritten: the
    `Reading input:` and `Logging to:` lines (`engine/mod.rs:55, 57`) name `name`'s input and log,
    and the timing lines are dropped. Throws if the result still names `dir`, the home directory
    or the host name.
    """
    function scrub(text::AbstractString, name::AbstractString, dir::AbstractString)
        stem = splitext(name)[1]
        lines = String[]
        for line in eachsplit(text, '\n'; keepempty = true)
            if startswith(line, "Reading input:")
                push!(lines, "Reading input: " * name)
            elseif startswith(line, "Logging to:")
                push!(lines, "Logging to: " * stem * ".log")
            elseif !occursin(TIMING, line)
                push!(lines, String(line))
            end
        end
        out = join(lines, '\n')
        for (what, s) in (("the run directory", dir), ("the run directory", realpath(dir)), ("the home directory", homedir()), ("the host name", gethostname()))
            isempty(s) && continue
            occursin(s, out) && error("oracle: the output of $name names $what")
        end
        return out
    end

    # ---------------------------------------------------------------- extractors

    _float(s::AbstractString) = parse(Float64, s)

    function _last_match(re::Regex, text::AbstractString, what::AbstractString, kind::AbstractString = "log")
        m = nothing
        for line in eachsplit(text, '\n')
            mm = match(re, line)
            mm === nothing || (m = mm)
        end
        m === nothing && error("oracle $kind: no `$what` line")
        return m
    end

    """
        total_energy(log) -> Float64

    The value of the last `  Total Energy:` line (`engine/results.rs:18`,
    `"  Total Energy:  {:20.12} Hartree"`). For MP2 and CCSD runs it is the SCF energy. The
    `SCF converged … E =` line excludes nuclear repulsion and is never used for it.
    """
    total_energy(log::AbstractString) =
        _float(_last_match(r"^  Total Energy:\s+(\S+) Hartree$", log, "Total Energy:")[1])

    """
        scf_iterations(log) -> Vector{Float64}

    The energies of the SCF iteration lines, for convergence-history checks only. A restricted
    run prints `SCF iter {:3}  E = {:16.12}  ΔE = {:8.3e}` with nuclear repulsion included
    (`scf/scf_cycle.rs:500-503`); an unrestricted run prints `UHF iter {:3}  E = {:18.12}  ΔE =
    {:.3e}` without it (`scf/uhf.rs:260`).
    """
    function scf_iterations(log::AbstractString)
        e = Float64[]
        for line in eachsplit(log, '\n')
            m = match(r"^(SCF|UHF) iter\s+\d+\s+E =\s+(\S+)\s+ΔE =", line)
            m === nothing || push!(e, _float(m[2]))
        end
        isempty(e) && error("oracle log: no `SCF iter` or `UHF iter` line")
        return e
    end

    """
        opt_steps(log) -> Vector{@NamedTuple{step::Int, energy::Float64, gnorm::Float64}}

    The `[OPT] Step` lines (`io/writer.rs:41`, `"[OPT] Step {:3}: E = {:17.10} |grad| = {:.3e}"`).
    """
    function opt_steps(log::AbstractString)
        steps = @NamedTuple{step::Int, energy::Float64, gnorm::Float64}[]
        for line in eachsplit(log, '\n')
            m = match(r"^\[OPT\] Step\s+(\d+): E =\s+(\S+) \|grad\| = (\S+)$", line)
            m === nothing || push!(steps, (step = parse(Int, m[1]), energy = _float(m[2]), gnorm = _float(m[3])))
        end
        isempty(steps) && error("oracle log: no `[OPT] Step` line")
        return steps
    end

    """
        mp2_correlation(stdout) -> Float64

    The last `Total MP2 Correlation:` line (`scf/mp2.rs:74`, `"  Total MP2 Correlation: {:.10}
    Hartree"`, printed with `println!`, so only in the standard output).
    """
    mp2_correlation(stdout::AbstractString) =
        _float(_last_match(r"^  Total MP2 Correlation: (\S+) Hartree$", stdout, "Total MP2 Correlation:", "stdout")[1])

    """
        ccsd_correlation(stdout) -> Float64

    The CCSD correlation energy without the triples correction: the last `FINAL CCSD
    CORRELATION:` line of the restricted CCSD driver (`scf/ccsd.rs:365`, `"  FINAL CCSD
    CORRELATION:  {:18.10} Hartree"`, printed with `println!`, so only in the standard output).
    """
    ccsd_correlation(stdout::AbstractString) =
        _float(_last_match(r"^  FINAL CCSD CORRELATION:\s+(\S+) Hartree$", stdout, "FINAL CCSD CORRELATION:", "stdout")[1])

    """
        frequencies(log) -> Vector{Float64}

    The vibrational frequencies, in cm⁻¹, of the last `VIBRATIONAL FREQUENCIES` table
    (`vibrations/mod.rs:154-178`; rows `"║ {:4} │ {:16.4} │ {:8} │ {:13.6} ║"`). The rows of type
    `T/R` (|ν| < 50 cm⁻¹, `vibrations/mod.rs:162`) are translations and rotations and are left
    out; the `Vib` and `Im` rows are returned in table order, an imaginary mode as a negative value.
    """
    function frequencies(log::AbstractString)
        lines = split(log, '\n')
        start = findlast(l -> occursin("VIBRATIONAL FREQUENCIES", l), lines)
        start === nothing && error("oracle log: no `VIBRATIONAL FREQUENCIES` line")
        ν = Float64[]
        for line in lines[(start + 1):end]
            startswith(line, "╚") && return ν
            m = match(r"^║\s+\d+ │\s+(\S+) │ (\S+)\s+│\s+\S+ ║$", line)
            m === nothing && continue
            m[2] == "T/R" || push!(ν, _float(m[1]))
        end
        error("oracle log: the `VIBRATIONAL FREQUENCIES` table has no closing line")
    end

    const _EXT_COUNT = r"^ Number of .* \[(NACP|BCP|RCP|CCP)\]:\s+(\d+)$"
    const _NUM = raw"(-?[0-9]+\.[0-9]+)"
    const _EXT_BCP = Regex("^ (\\d+)\\s+\\S+ - \\S+\\s+" * join(fill(_NUM, 6), "\\s+") * "\$")
    const _EXT_BCP_KEYS = ("r_e", "r_b", "delta_r", "rho", "laplacian", "ellipticity")

    """
        ext_report(path) -> Dict{String,Float64}

    The critical-point report the oracle's `aimext` command writes (`analysis/aim.rs:198-336`):
    the counts of section 2 (`aim.rs:226-229`) as `n_nacp`, `n_bcp`, `n_rcp` and `n_ccp`, and for
    each row `k` of the bond-critical-point table of section 3 (`aim.rs:252-255`, `" {:<4}  {:<14}
    {:10.5}   {:10.5}   {:10.5}   {:11.6}  {:11.6}  {:10.5}"`) the keys `bcp<k>.r_e`,
    `bcp<k>.r_b`, `bcp<k>.delta_r` (bohr), `bcp<k>.rho`, `bcp<k>.laplacian` (a.u.) and
    `bcp<k>.ellipticity`.
    """
    function ext_report(path::AbstractString)
        d = Dict{String, Float64}()
        for line in eachline(path)
            m = match(_EXT_COUNT, line)
            if m !== nothing
                d["n_" * lowercase(m[1])] = parse(Int, m[2])
                continue
            end
            m = match(_EXT_BCP, line)
            m === nothing && continue
            for (i, k) in enumerate(_EXT_BCP_KEYS)
                d["bcp" * m[1] * "." * k] = _float(m[i + 1])
            end
        end
        all(haskey(d, k) for k in ("n_nacp", "n_bcp", "n_rcp", "n_ccp")) ||
            error("oracle ext: no critical-point counts in $(basename(path))")
        return d
    end

    """
        nuclear_repulsion_from(log) -> Float64

    The nuclear repulsion of a closed-shell run: the last `Total Energy:` value
    (`engine/results.rs:18`, 12 decimals) minus the last `SCF converged … E =` value, which
    excludes it (`scf/scf_cycle.rs:591`, `{:.12}`). An unrestricted run prints no such line
    (`scf/uhf.rs:302`), so this throws for it.
    """
    function nuclear_repulsion_from(log::AbstractString)
        e_scf = _float(_last_match(r"^SCF converged in \d+ iterations  E = (\S+)$", log, "SCF converged … E =")[1])
        return total_energy(log) - e_scf
    end

    """
        echoed_settings(stdout) -> Dict{String,Any}

    The settings the oracle echoes in its `Calculation Summary` (`engine/mod.rs:76-88`): `title`,
    `method` (the oracle's own method name, `engine/config.rs:252`), `basis`, `multiplicity`,
    `charge` and `convergence` (Rust's `{:.1e}`, kept as text), and `atoms`, the first `  Atoms:`
    line (`engine/electronic.rs:24, 279`), when the run mode prints one.
    """
    function echoed_settings(stdout::AbstractString)
        lines = split(stdout, '\n')
        start = findfirst(==("║  CALCULATION SUMMARY  ║"), lines)
        start === nothing && error("oracle stdout: no `Calculation Summary` heading")
        fields = (
            "Title" => "title", "Method" => "method", "Basis Set" => "basis", "Multiplicity" => "multiplicity",
            "Charge" => "charge", "Convergence" => "convergence",
        )
        d = Dict{String, Any}()
        for line in lines[(start + 1):end], (label, key) in fields
            haskey(d, key) && continue
            m = match(Regex("^" * label * ":\\s+(.*)\$"), line)
            m === nothing && continue
            d[key] = key in ("multiplicity", "charge") ? parse(Int, m[1]) : String(m[1])
        end
        for (_, key) in fields
            haskey(d, key) || error("oracle stdout: no `$key` line in the Calculation Summary")
        end
        for line in lines[(start + 1):end]
            m = match(r"^  Atoms:\s+(\d+)$", line)
            m === nothing && continue
            d["atoms"] = parse(Int, m[1])
            break
        end
        return d
    end

    # ---------------------------------------------------------------- stored data

    references() = TOML.parsefile(REFERENCES)

    function case_table(case::AbstractString, refs = references())
        cases = get(refs, "case", Dict{String, Any}())
        haskey(cases, case) || error("oracle: no case $case in references.toml")
        return cases[case]
    end

    """
        case_names() -> Vector{String}

    The names of the `[case]` tables in `references.toml`, sorted.
    """
    case_names() = sort!(collect(keys(get(references(), "case", Dict{String, Any}()))))

    """
        case_input(case) -> String

    The input text of `case`.
    """
    case_input(case::AbstractString) = case_table(case)["input"]::String

    """
        reference(case, key) -> Float64

    The stored value `key` of `case` in `references.toml`.
    """
    function reference(case::AbstractString, key::AbstractString)
        t = case_table(case)
        v = get(t, key, nothing)
        v isa Float64 || error("oracle: no reference value $key for $case in references.toml")
        return v
    end

    """
        reference_vector(case, key) -> Vector{Float64}

    The stored list of values `key` of `case` in `references.toml`.
    """
    function reference_vector(case::AbstractString, key::AbstractString)
        v = get(case_table(case), key, nothing)
        v isa Vector && all(x -> x isa Float64, v) ||
            error("oracle: no reference list $key for $case in references.toml")
        return Vector{Float64}(v)
    end

    """
        tolerance(case, key) -> Float64

    The parity tolerance of the quantity `key` of `case`: the row its `tolerance` map names, read
    from `tolerances.toml`. A quantity with no entry in the map, or an entry naming a row that does
    not exist, throws: no row, no comparison.
    """
    function tolerance(case::AbstractString, key::AbstractString)
        row = get(get(case_table(case), "tolerance", Dict{String, Any}()), key, nothing)
        row === nothing && error("oracle: no tolerance row (no map entry) for $case.$key")
        t = TOML.parsefile(TOLERANCES)
        for part in split(row, '.')
            t isa Dict && haskey(t, part) || error("oracle: no tolerance row $row for $case.$key")
            t = t[part]
        end
        t isa Dict && t["value"] isa Float64 || error("oracle: no tolerance row $row for $case.$key")
        return t["value"]::Float64
    end

    """
        quantity_extractor(key) -> String

    The name of the extractor that gives the quantity `key`.
    """
    function quantity_extractor(key::AbstractString)
        key in ("total_energy", "mp2_correlation", "ccsd_correlation", "frequencies") && return String(key)
        key == "nuclear_repulsion" && return "nuclear_repulsion_from"
        key in ("opt_energy", "opt_gnorm") && return "opt_steps"
        startswith(key, "ext_report.") && return "ext_report"
        error("oracle: no extractor for the quantity $key")
    end

    """
        quantity(case, key, log, stdout, dir) -> Union{Float64,Vector{Float64}}

    The quantity `key` of `case`, read from the run's `log` and `stdout`, and from the files it
    exported, named `<case>.<extension>` in `dir`. `opt_energy` and `opt_gnorm` are the last
    `[OPT] Step` line's; `ext_report.<k>` is the key `k` of `<case>.ext`.
    """
    function quantity(case::AbstractString, key::AbstractString, log::AbstractString, stdout::AbstractString, dir::AbstractString)
        key == "total_energy" && return total_energy(log)
        key == "nuclear_repulsion" && return nuclear_repulsion_from(log)
        key == "mp2_correlation" && return mp2_correlation(stdout)
        key == "ccsd_correlation" && return ccsd_correlation(stdout)
        key == "opt_energy" && return last(opt_steps(log)).energy
        key == "opt_gnorm" && return last(opt_steps(log)).gnorm
        key == "frequencies" && return frequencies(log)
        if startswith(key, "ext_report.")
            d = ext_report(joinpath(dir, case * ".ext"))
            k = chopprefix(key, "ext_report.")
            haskey(d, k) || error("oracle ext: no key $k in $case.ext")
            return d[k]
        end
        error("oracle: no extractor for the quantity $key")
    end

    # ---------------------------------------------------------------- writing references.toml

    const HEADER = """
    # Oracle reference values. Each [case] table holds an input, its source, the oracle command
    # that runs it (`run` or `aimext`), the extractors and exports, and the map from each compared
    # quantity to its row in tolerances.toml; capture.jl writes the extracted values and the
    # SHA-256 of each captured file. Each [fixture] table holds the settings the oracle echoes for a
    # vendored input file. Rewritten by capture.jl.
    """

    const _CASE_HEAD = ("input", "source", "command", "extract", "exports", "tolerance")

    _key(k::AbstractString) = occursin(r"^[A-Za-z0-9_-]+$", k) ? String(k) : _value(k)

    function _value(v)
        v isa Float64 && !isfinite(v) && error("oracle: references.toml cannot hold the value $v")
        v isa Vector{Any} && isempty(v) && return "[]"
        s = sprint(TOML.print, Dict("v" => v))
        return chopsuffix(chopprefix(s, "v = "), "\n")
    end

    _inline(d::AbstractDict) = "{" * join([_key(k) * " = " * _value(d[k]) for k in sort!(collect(keys(d)))], ", ") * "}"

    function _multiline_literal(s::AbstractString)
        occursin("'''", s) && error("oracle: an input holds ''' and cannot be a TOML multiline literal")
        any(c -> iscntrl(c) && c != '\n' && c != '\t', s) &&
            error("oracle: an input holds a control character and cannot be a TOML multiline literal")
        return "'''\n" * s * "'''"
    end

    """
        write_references(io, refs)

    Write the references `refs` (the parsed `references.toml`) in the file's fixed form: the
    header, `[meta]`, each `[case]` table with its input as a multiline literal, then its other
    keys and its `sha256` subtable, then each `[fixture]` table.
    """
    function write_references(io::Base.IO, refs::AbstractDict)
        print(io, HEADER)
        println(io)
        println(io, "[meta]")
        for (k, v) in sort!(collect(get(refs, "meta", Dict{String, Any}())); by = first)
            println(io, _key(k), " = ", _value(v))
        end
        cases = get(refs, "case", Dict{String, Any}())
        for name in sort!(collect(keys(cases)))
            t = cases[name]
            println(io)
            println(io, "[case.", _key(name), "]")
            for k in _CASE_HEAD
                haskey(t, k) || error("oracle: case $name has no `$k`")
                v = t[k]
                s = k == "input" ? _multiline_literal(v) : k == "tolerance" ? _inline(v) : _value(v)
                println(io, k, " = ", s)
            end
            subtables = String[]
            for k in sort!(collect(keys(t)))
                k in _CASE_HEAD && continue
                t[k] isa AbstractDict ? push!(subtables, k) : println(io, _key(k), " = ", _value(t[k]))
            end
            for k in subtables
                println(io)
                println(io, "[case.", _key(name), ".", _key(k), "]")
                for (kk, vv) in sort!(collect(t[k]); by = first)
                    println(io, _key(kk), " = ", _value(vv))
                end
            end
        end
        fixtures = get(refs, "fixture", Dict{String, Any}())
        for name in sort!(collect(keys(fixtures)))
            println(io)
            println(io, "[fixture.", _value(name), "]")
            for (k, v) in sort!(collect(fixtures[name]); by = first)
                println(io, _key(k), " = ", _value(v))
            end
        end
        return nothing
    end
end
