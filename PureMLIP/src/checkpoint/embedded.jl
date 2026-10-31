# Archives stored as byte strings inside a checkpoint's pickle.

const ZIP_LOCAL_MAGIC = UInt8[0x50, 0x4b, 0x03, 0x04]   # "PK\x03\x04"

"""
    embedded_archives(t::PickleTree) -> Vector{TorchArchive}

One archive for each `PK_BYTES` node of `t` whose bytes begin with a ZIP local-file
signature, in node order.
"""
function embedded_archives(t::PickleTree)
    out = TorchArchive[]
    for node in t.nodes
        node.kind == PK_BYTES || continue
        b = codeunits(node.str)
        length(b) >= 4 && view(b, 1:4) == ZIP_LOCAL_MAGIC || continue
        push!(out, open_torch_archive(Vector{UInt8}(b)))
    end
    return out
end
