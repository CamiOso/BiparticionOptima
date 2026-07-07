module TpmIO

using Mmap

export mmap_npy

function _parse_header(path::String)
    open(path, "r") do f
        magic = read(f, 6)
        magic == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] ||
            error("No es un archivo .npy válido: $path")

        major = read(f, UInt8)
        _minor = read(f, UInt8)

        hlen = if major == 1
            Int(read(f, UInt16))
        else
            Int(read(f, UInt32))
        end

        header = String(read(f, hlen))
        data_offset = position(f)

        # dtype
        dm = match(r"'descr':\s*'([^']+)'", header)
        dm === nothing && error("No se pudo parsear dtype: $header")
        dtype_str = dm.captures[1]
        T = if dtype_str in ("<f4", "=f4", "|f4")
            Float32
        elseif dtype_str in ("<f8", "=f8", "|f8")
            Float64
        elseif dtype_str in ("<f2", "=f2", "|f2")
            Float16
        else
            error("Dtype no soportado: $dtype_str")
        end

        # shape
        sm = match(r"'shape':\s*\(([^)]*)\)", header)
        sm === nothing && error("No se pudo parsear shape: $header")
        dims = [parse(Int, strip(s)) for s in split(sm.captures[1], ',') if !isempty(strip(s))]

        fortran = occursin("'fortran_order': True", header)

        data_offset, T, dims, fortran
    end
end

"""
    mmap_npy(path) -> Matrix{T}

Lee un archivo .npy 2D y lo retorna como matriz Julia memory-mapped.
Para arrays C-order (numpy), retorna shape (ncols, nrows) para que
tpm_jl[nodo+1, estado+1] coincida con tpm_py[estado, nodo].
"""
function mmap_npy(path::String)
    offset, T, dims, fortran = _parse_header(path)
    length(dims) == 2 || error("Solo se soportan arrays 2D, shape=$dims")
    nrows, ncols = dims[1], dims[2]

    # C-order (numpy default): filas contiguas → en Julia (column-major)
    # lo mapeamos como (ncols, nrows) para que el indexado sea correcto.
    if !fortran
        Mmap.mmap(path, Matrix{T}, (ncols, nrows), offset)
    else
        Mmap.mmap(path, Matrix{T}, (nrows, ncols), offset)
    end
end

end # module
