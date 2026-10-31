# Precompile workload: the entry points whose first call dominates cold-start latency.
# It runs in memory only; the checkpoint reader is not exercised because building a checkpoint
# needs a ZIP and pickle writer, which lives in the tests.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    v = [0.1, 0.2, 0.9]
    R = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]
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
    end
end
