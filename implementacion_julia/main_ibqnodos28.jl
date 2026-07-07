"""
IBQNodos para n=28 en Julia — versión memory-efficient.

Ventajas vs Python:
  • La TPM de ~30 GB se memory-mapea (sin copia en RAM)
  • Los perfiles IB usan 2048 muestras en vez de 268M → sin OOM
  • marginal_directo suma solo 2^(n_free) estados, no materializa tensores

Uso:
    julia -t auto --project=. main_ibqnodos28.jl            # usa todos los cores (recomendado)
    julia -t auto --project=. main_ibqnodos28.jl --desde 5  # retomar desde caso 5
    julia --project=. main_ibqnodos28.jl                     # 1 thread (sin paralelismo LUT)
"""

using Pkg
Pkg.activate(@__DIR__)

include("src/tpm_io.jl")
include("src/ibqnodos.jl")
using .TpmIO, .IBQNodos
using JSON3, Dates

# ─── Configuración ────────────────────────────────────────────────────────────
const N          = 28
const SISTEMA    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ01"   # 26 letras + '0' + '1'
const ESTADO     = "1" * "0"^(N - 1)    # "100...0"  (nodo A encendido)
const CONDICION  = "1"^N                 # sin condicionamiento

const PROJ_DIR  = dirname(abspath(@__FILE__))
const TPM_PATH  = joinpath(PROJ_DIR, "..", "src", ".samples", "N28A.npy")
const CKPT_PATH = joinpath(PROJ_DIR, "checkpoint_ibqnodos_28a.json")

# ─── Helpers ──────────────────────────────────────────────────────────────────
function to_mask(letters::AbstractString)::String
    s = Set(uppercase(c) for c in letters)
    join(c ∈ s ? '1' : '0' for c in SISTEMA)
end

# ─── Checkpoint ───────────────────────────────────────────────────────────────
function cargar_checkpoint()::Dict
    isfile(CKPT_PATH) ? JSON3.read(read(CKPT_PATH, String), Dict) : Dict("casos" => [])
end

function guardar_checkpoint(data::Dict)
    write(CKPT_PATH, JSON3.write(data))
end

# ─── Leer casos desde JSON ────────────────────────────────────────────────────
const CASOS_JSON = joinpath(PROJ_DIR, "casos_28a.json")

function leer_casos()::Vector{Dict}
    isfile(CASOS_JSON) || error("No se encontró $CASOS_JSON.\nGenerarlo con: python3 implementacion_julia/extraer_casos_28a.py")
    raw = JSON3.read(read(CASOS_JSON, String), Vector)
    [Dict(
        "fila"    => Int(c["prueba"]) + 5,
        "alc_str" => String(c["alc"]),
        "mec_str" => String(c["mec"]),
        "alc_bin" => to_mask(String(c["alc"])),
        "mec_bin" => to_mask(String(c["mec"])),
    ) for c in raw]
end

function inicializar_hoja_resultados()
    println("  [INFO] Resultados → checkpoint JSON + resultados_28a.json")
end

function escribir_resultado(fila::Int, prueba::Int, alc_str::String, mec_str::String,
                             phi::Union{Float32, Nothing}, t::Float64, particion::String)
end

# ─── Main ─────────────────────────────────────────────────────────────────────
function main()
    desde = 0
    for (i, arg) in enumerate(ARGS)
        if arg == "--desde" && i < length(ARGS)
            desde = parse(Int, ARGS[i+1])
        end
    end

    tpm_path = abspath(TPM_PATH)
    isfile(tpm_path) || error("TPM no encontrada: $tpm_path\nEjecuta primero: python scripts/generar_N28A.py")

    sz_gb = filesize(tpm_path) / 1e9
    println("Cargando TPM $(basename(tpm_path)) ($(round(sz_gb, digits=2)) GB)..."); flush(stdout)
    t0 = time()
    tpm_jl = TpmIO.mmap_npy(tpm_path)
    println("TPM mapeada en $(round(time()-t0, digits=1))s | shape=$(reverse(size(tpm_jl))) (n_nodos×n_estados)"); flush(stdout)
    println("Uso RAM: ~0 MB (memory-mapped, OS pagea bajo demanda)"); flush(stdout)

    inicializar_hoja_resultados()
    casos      = leer_casos()
    ckpt       = cargar_checkpoint()
    done_filas = Set(c["fila"] for c in get(ckpt, "casos", []) if haskey(c, "phi"))

    seed_cache = Dict{String, Tuple{Set{Int}, Set{Int}}}()
    for c in get(ckpt, "casos", [])
        if haskey(c, "phi") && haskey(c, "alc_A") && haskey(c, "mec_A")
            seed_cache[c["mec_bin"]] = (Set{Int}(c["alc_A"]), Set{Int}(c["mec_A"]))
        end
    end

    resultados = copy(get(ckpt, "casos", []))

    println("\n$("="^60)")
    println("  28A Julia: $(length(casos)) casos | completados: $(length(done_filas))")
    println("$("="^60)\n")

    t_total = time()

    for (i, caso) in enumerate(casos)
        caso["fila"] < desde && continue
        caso["fila"] ∈ done_filas && begin
            println("  [$i/$(length(casos))] fila=$(caso["fila"]) — ya completado, saltando.")
            continue
        end

        n_alc = count(==('1'), caso["alc_bin"])
        n_mec = count(==('1'), caso["mec_bin"])
        println("  [$i/$(length(casos))] fila=$(caso["fila"]) alc=$(caso["alc_str"][1:min(10,end)])...($n_alc) mec=$(caso["mec_str"][1:min(10,end)])...($n_mec)"); flush(stdout)

        warm = get(seed_cache, caso["mec_bin"], nothing)
        warm !== nothing && println("    warm-start: |alc_A|=$(length(warm[1])), |mec_A|=$(length(warm[2]))")

        t0_caso = time()
        phi_val   = nothing
        particion = "ERROR"
        new_alc_A = Set{Int}()
        new_mec_A = Set{Int}()

        try
            warm_alc = warm !== nothing ? warm[1] : nothing
            warm_mec = warm !== nothing ? warm[2] : nothing

            phi_f32, particion, new_alc_A, new_mec_A = ibqnodos_caso(
                tpm_jl,
                caso["alc_bin"],
                caso["mec_bin"],
                ESTADO;
                warm_alc_A = warm_alc,
                warm_mec_A = warm_mec,
                n_total    = N,
                verbose    = true
            )
            phi_val = phi_f32
            seed_cache[caso["mec_bin"]] = (new_alc_A, new_mec_A)
            println("    φ=$(round(Float64(phi_f32), digits=6))  t=$(round(time()-t0_caso, digits=1))s")
        catch ex
            println("    ERROR: $ex")
            particion = "ERROR: $ex"
        end

        elapsed = time() - t0_caso
        flush(stdout)

        caso_res = merge(caso, Dict(
            "phi"       => phi_val !== nothing ? Float64(phi_val) : nothing,
            "t"         => round(elapsed, digits=3),
            "particion" => particion,
            "alc_A"     => sort(collect(new_alc_A)),
            "mec_A"     => sort(collect(new_mec_A)),
        ))

        idx = findfirst(c -> c["fila"] == caso["fila"], resultados)
        idx !== nothing ? (resultados[idx] = caso_res) : push!(resultados, caso_res)
        guardar_checkpoint(Dict("casos" => resultados))
    end

    t_fin = round(time() - t_total, digits=1)
    completados = count(c -> haskey(c, "phi") && c["phi"] !== nothing, resultados)
    println("\n$("="^60)")
    println("  Finalizados: $completados/$(length(casos))  |  Tiempo total: $(t_fin)s")
    println("$("="^60)")

    res_path = joinpath(PROJ_DIR, "resultados_28a.json")
    write(res_path, JSON3.write(Dict("casos" => resultados)))
    println("  Resultados guardados en: $res_path")
    println("  Para actualizar el Excel: python3 actualizar_excel_28a.py")
end

main()
