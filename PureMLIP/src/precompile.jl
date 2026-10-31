# Precompile workload: the entry points whose first call dominates cold-start latency.
# It runs in memory only: the model path reads a tiny checkpoint written in memory by
# `tiny_mace_checkpoint`, so it opens no file and uses no network.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    v = [0.1, 0.2, 0.9]
    R = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]
    P = [0.0 1.4 0.2; 0.0 0.1 1.3; 0.0 0.3 0.1]
    cell = [6.5 0.0 0.0; 0.4 6.2 0.0; 0.3 0.2 6.8]
    @compile_workload begin
        Y = zeros(sh_length(3))
        dY = zeros(sh_length(3), 3)
        spherical_harmonics!(Y, v, 3)
        spherical_harmonics_and_gradient!(Y, dY, v, 3)
        e3nn_spherical_harmonics!(Y, v, 3)
        e3nn_spherical_harmonics_and_gradient!(Y, dY, v, 3)
        real_clebsch_gordan(1, 1, 2)
        e3nn_wigner_3j(1, 1, 2)
        e3nn_wigner_3j(1, 2, 3)
        wigner_D(2, R)
        ck = load_mace_checkpoint(tiny_mace_checkpoint())
        model = MACEModel{Float64}(ck)
        mace_energy_forces(model, P, cell, [29, 1, 29])
    end
end
