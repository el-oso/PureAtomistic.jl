@testitem "layer graph holds" tags = [:checks] begin
    using StrictModeTest
    test_layers(
        PureMLIP => (PureMLIP.Checkpoint, PureMLIP.Equivariant),
        PureMLIP.Checkpoint => (),
        PureMLIP.Equivariant => ()
    )
end

# An independent second check of the no-private-access rule between PureMLIP's submodules,
# next to `test_layers` above. `allow_internal_accesses = false` keeps reads between modules of
# one package; `from = (PureMLIP,)` judges only names owned by PureMLIP and its submodules, so
# reads of Base, Core and other packages are out of scope. Only the package source is parsed,
# not the test files.
@testitem "submodules read only each other's public names" tags = [:checks] begin
    using ExplicitImports
    for M in (PureMLIP, PureMLIP.Checkpoint, PureMLIP.Equivariant)
        @test isnothing(
            check_all_qualified_accesses_are_public(
                M, pathof(PureMLIP); from = (PureMLIP,), allow_internal_accesses = false
            )
        )
    end
end

# `test_calls(M)` sweeps the signatures compiled for functions whose parent module is `M`, so
# each submodule is swept on its own, after its entry points have run; `PureMLIP` itself owns
# no functions, so its own sweep is empty. A sweep that checked nothing would pass vacuously;
# the count guards against that.
@testitem "every compiled signature resolves (test_calls)" tags = [:checks] setup = [Checkpoints] begin
    using StrictModeTest
    ck = load_mace_checkpoint(Checkpoints.fetch(:mp0b3))
    torch_array(first(values(ck.state.f64)), Val(1))
    a = PureMLIP.open_torch_archive(read(Checkpoints.fetch(:mp0b3)))
    PureMLIP.embedded_archives(PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl")))

    v = [0.1, 0.2, 0.9]
    Y = zeros(sh_length(3)); dY = zeros(sh_length(3), 3)
    spherical_harmonics!(Y, v, 3); spherical_harmonics_and_gradient!(Y, dY, v, 3)
    e3nn_spherical_harmonics!(Y, v, 3); e3nn_spherical_harmonics_and_gradient!(Y, dY, v, 3)
    sh_index(2, 1); clebsch_gordan_complex(1, 0, 1, 0, 2, 0)
    real_clebsch_gordan(1, 1, 2); e3nn_wigner_3j(1, 1, 2); e3nn_basis_change(2)
    wigner_D(2, [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])

    test_calls(PureMLIP)
    for M in (PureMLIP.Checkpoint, PureMLIP.Equivariant)
        @test length(test_calls(M)) > 0
    end
end

@testitem "Aqua" tags = [:checks] begin
    using Aqua
    Aqua.test_all(PureMLIP)
end
