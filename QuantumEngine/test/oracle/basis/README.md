# The oracle's placeholder auxiliary basis

`oracle-def2-svp-jkfit.gbs` is the file `basis_sets/def2-svp-jkfit.gbs` of the
repository https://github.com/boleklolek/quantum_engine at commit
4a78a1d235688c5af10ccd94be75d7292073ff8d, copied byte for byte under a new
name. It is the oracle's test placeholder, not a published basis set: its
first line reads `# def2-SVP-JKFIT auxiliary basis (simplified for testing)`,
and it holds H, C, N and O only. The oracle reads it as its default auxiliary
basis `def2-SVP-JKFIT`.

QuantumEngine resolves the name `def2-svp-jkfit` to the shipped
`def2-universal-jkfit`, so no basis name reaches this file through the default
search path. Its stem `oracle-def2-svp-jkfit` is in no alias table. The file is
kept so that a regression test can build the oracle's actual auxiliary basis
with

    build_basis(mol, "oracle-def2-svp-jkfit"; paths = [joinpath(pkgdir(QuantumEngine), "test", "oracle", "basis")])

The source repository declares the Apache-2.0 license in its `Cargo.toml`; the
license text is in `../inputs/LICENSE`.
