# Basis names: the alias table, the name normalization and the candidate file stems.

# The oracle's alias groups (`reader.rs:54-123`), transcribed as data: every name of a group maps
# to the group's file stem. Four rows differ from the oracle and say so.
const _ALIAS_GROUPS = Pair{Vector{String}, String}[
    # Pople
    ["sto3g", "sto-3g"] => "sto-3g",
    ["321g", "3-21-g", "3-21g"] => "3-21g",
    ["431g", "4-31-g", "4-31g"] => "4-31g",
    ["621g", "6-21-g", "6-21g"] => "6-21g",
    ["631-g", "631g", "6-31-g", "6-31g"] => "6-31g",
    ["631g*", "631-g*", "6-31g*", "6-31g(d)", "631g(d)"] => "6-31g_star",
    # The oracle maps this group to `6-31g_dp`, a file with no Basis Set Exchange header whose O d
    # exponent differs from the Basis Set Exchange file named here.
    ["631g**", "631-g**", "6-31g**", "6-31g(d,p)", "631g(d,p)"] => "6-31g_star_star",
    ["631+g", "6-31+g", "6-31plusg"] => "6-31plusg",
    ["631+g*", "6-31+g*", "6-31+g(d)", "631+g(d)", "6-31plusg*"] => "6-31plusg_star",
    ["631++g**", "6-31++g**", "6-31++g(d,p)", "631++g(d,p)", "6-31plusplusg**"] => "6-31plusplusg_star_star",
    ["6311g", "6311-g", "6-311-g", "6-311g"] => "6-311g",
    ["6311g*", "6311-g*", "6-311g*", "6-311g(d)", "6311g(d)"] => "6-311g_star",
    ["6311g**", "6311-g**", "6-311g**", "6-311g(d,p)", "6311g(d,p)"] => "6-311g_star_star",
    ["6311+g*", "6-311+g*", "6-311+g(d)", "6311+g(d)"] => "6-311plusg_star",
    # The oracle maps this group to `6-311plusplusg_dp`, a file with no Basis Set Exchange header
    # whose C, N, O, F and Ne blocks have one shell fewer than the Basis Set Exchange file named here.
    ["6311++g(d,p)", "6311++g**", "6-311++g**", "6-311++g(d,p)"] => "6-311plusplusg_star_star",
    # The oracle maps this group to `6-311++g_2d2p`, a file with no Basis Set Exchange header that
    # holds only H, C and N, identical to their blocks in the Basis Set Exchange file named here.
    ["6311++g2d2p", "6-311++g(2d,2p)", "6311++2d2p"] => "6-311plusplusg(2d,2p)",
    ["6311++g3df3pd", "6-311++g(3df,3pd)"] => "6-311plusplusg(3df,3pd)",
    # Ahlrichs and Weigend (def2)
    ["def2svp", "def2-svp"] => "def2-svp",
    ["def2svpd", "def2-svpd"] => "def2-svpd",
    ["def2sv(p)", "def2-sv(p)"] => "def2-sv(p)",
    ["def2tzvp", "def2-tzvp"] => "def2-tzvp",
    ["def2tzvpd", "def2-tzvpd"] => "def2-tzvpd",
    ["def2tzvpp", "def2-tzvpp"] => "def2-tzvpp",
    ["def2tzvppd", "def2-tzvppd"] => "def2-tzvppd",
    ["def2qzvp", "def2-qzvp"] => "def2-qzvp",
    ["def2qzvpd", "def2-qzvpd"] => "def2-qzvpd",
    ["def2qzvpp", "def2-qzvpp"] => "def2-qzvpp",
    ["def2qzvppd", "def2-qzvppd"] => "def2-qzvppd",
    ["def2jkfit", "def2-universal-jkfit"] => "def2-universal-jkfit",
    # Not in the oracle's table, which passes `def2-svp-jkfit` through to its own placeholder file
    # (H, C, N and O only): the name is the default auxiliary basis, read as the universal JK fit.
    ["def2-svp-jkfit"] => "def2-universal-jkfit",
    # Dunning correlation consistent
    ["ccpvdz", "cc-pvdz"] => "cc-pvdz",
    ["ccpvtz", "cc-pvtz"] => "cc-pvtz",
    ["ccpvqz", "cc-pvqz"] => "cc-pvqz",
    ["ccpv5z", "cc-pv5z"] => "cc-pv5z",
    ["ccpv6z", "cc-pv6z"] => "cc-pv6z",
    ["augccpvdz", "aug-cc-pvdz"] => "aug-cc-pvdz",
    ["augccpvtz", "aug-cc-pvtz"] => "aug-cc-pvtz",
    ["augccpvqz", "aug-cc-pvqz"] => "aug-cc-pvqz",
    ["augccpv5z", "aug-cc-pv5z"] => "aug-cc-pv5z",
    # Explicitly correlated (F12)
    ["ccpvdzf12", "cc-pvdz-f12"] => "cc-pvdz-f12",
    ["ccpvtzf12", "cc-pvtz-f12"] => "cc-pvtz-f12",
    ["ccpvqzf12", "cc-pvqz-f12"] => "cc-pvqz-f12",
    ["ccpvdzf12optri", "cc-pvdz-f12-optri"] => "cc-pvdz-f12-optri",
    ["ccpvtzf12optri", "cc-pvtz-f12-optri"] => "cc-pvtz-f12-optri",
    # Jensen polarization consistent
    ["pcseg0", "pcseg-0"] => "pcseg-0",
    ["pcseg1", "pcseg-1"] => "pcseg-1",
    ["pcseg2", "pcseg-2"] => "pcseg-2",
    ["pcseg3", "pcseg-3"] => "pcseg-3",
    ["pcseg4", "pcseg-4"] => "pcseg-4",
    ["augpcseg0", "aug-pcseg-0"] => "aug-pcseg-0",
    ["augpcseg1", "aug-pcseg-1"] => "aug-pcseg-1",
    ["augpcseg2", "aug-pcseg-2"] => "aug-pcseg-2",
    ["augpcseg3", "aug-pcseg-3"] => "aug-pcseg-3",
    ["pcsseg1", "pcsseg-1"] => "pcsseg-1",
    ["pcsseg2", "pcsseg-2"] => "pcsseg-2",
]

# Each lowercase alias to its file stem. No name occurs in two groups (an item checks it), and the
# table is idempotent on its values, which `candidate_stems` relies on.
const BASIS_ALIASES = let table = Dict{String, String}()
    for (names, stem) in _ALIAS_GROUPS, name in names
        table[name] = stem
    end
    table
end

"""
    normalize_basis_name(name::AbstractString) -> String

The file stem of the basis name `name`: `name` stripped of surrounding whitespace and lowercased,
then mapped through the alias table, so that `"6-31G*"` gives `"6-31g_star"`, `"631-g"` gives
`"6-31g"` and `"def2-SVP-JKFIT"` gives `"def2-universal-jkfit"`. A name not in the table is
returned stripped and lowercased. The table maps each of its values to itself.
"""
function normalize_basis_name(name::AbstractString)
    r = lowercase(strip(name))
    return get(BASIS_ALIASES, r, r)
end

# The file stems tried for the already-normalized name `s`, in order: `s` stripped and lowercased,
# then that name with `*` and `**` spelled out, with `**` as `_dp`, with `+` and `++` spelled out as
# well, and with `(d,p)` and `(d)` spelled out; a repeated stem is dropped, keeping the first. These
# are the oracle's spellings (`reader.rs:126-160`); no alias table is applied here.
function stem_variants(s::AbstractString)
    r = lowercase(strip(s))
    stems = String[r]
    for v in (
            replace(replace(r, "**" => "_star_star"), "*" => "_star"),
            replace(replace(r, "**" => "_dp"), "*" => "_star"),
            replace(replace(replace(replace(r, "++" => "plusplus"), "+" => "plus"), "**" => "_star_star"), "*" => "_star"),
            replace(replace(r, "(d,p)" => "_dp"), "(d)" => "_star"),
        )
        v in stems || push!(stems, v)
    end
    return stems
end

"""
    candidate_stems(name::AbstractString) -> Vector{String}

The file stems tried for the basis name `name`, in order and without repeats: first
`normalize_basis_name(name)`, then its spellings with `*` and `**` written as `_star`,
`_star_star` or `_dp`, with `+` and `++` written as `plus` and `plusplus`, and with `(d,p)` and
`(d)` written as `_dp` and `_star`. `candidate_stems("6-31+G**")` is
`["6-31+g**", "6-31+g_star_star", "6-31+g_dp", "6-31plusg_star_star"]`.
"""
candidate_stems(name::AbstractString) = stem_variants(normalize_basis_name(name))
