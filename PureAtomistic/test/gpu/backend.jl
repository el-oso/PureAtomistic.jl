# The backend that the :gpu items run on, named by PUREMLIP_GPU_BACKEND: CPU (KernelAbstractions'
# own CPU backend, no vendor package), Metal, CUDA or AMDGPU. Run the items with
#   PUREMLIP_GPU_BACKEND=<name> julia --check-bounds=yes --project=<pkg>/test/gpu/<name> <pkg>/test/gpu/runtests.jl
# On macOS, KernelAbstractions 0.10's CPU backend (PoCL, through hwloc) hangs at topology detection
# on Apple M-series chips with performance levels. HWLOC_COMPONENTS=-darwin makes hwloc skip its
# Darwin component; export it before instantiating or precompiling any environment that holds
# KernelAbstractions 0.10, and before launching the CPU backend.
@testmodule GPUBackend begin
    using KernelAbstractions

    const NAME = get(ENV, "PUREMLIP_GPU_BACKEND", "")
    NAME in ("CPU", "Metal", "CUDA", "AMDGPU") ||
        error("PUREMLIP_GPU_BACKEND must be CPU, Metal, CUDA or AMDGPU, got \"$NAME\"")

    if NAME == "CPU"
        const backend = KernelAbstractions.CPU()
        const isgpu = false
        const precisions = (Float32, Float64)
    else
        @eval using $(Symbol(NAME))
        const vendor = getfield(@__MODULE__, Symbol(NAME))
        vendor.functional() || error("$NAME is installed but not functional on this machine")
        const backend = NAME == "Metal" ? vendor.MetalBackend() :
            NAME == "CUDA" ? vendor.CUDABackend() : vendor.ROCBackend()
        const isgpu = true
        # Metal has no Float64.
        const precisions = NAME == "Metal" ? (Float32,) : (Float32, Float64)
    end

    # A copy of `A` allocated on the backend.
    function todevice(A::AbstractArray)
        D = KernelAbstractions.allocate(backend, eltype(A), size(A))
        copyto!(D, A)
        return D
    end

    export backend, isgpu, precisions, todevice
end
