module IBQNodos

using Statistics, Random, LinearAlgebra

export ibqnodos_caso, fmt_biparticion

# ─── Constantes ────────────────────────────────────────────────────────────────
const N_PROFILE_SAMPLES = 2048
const BETA_IB            = 4f0
const MAX_ITER_IB        = 60
const N_RESTARTS_IB      = 3
const TEMP_INICIAL       = 1f0
const TEMP_FINAL         = 1f-3
const FACTOR_ENFRIAM     = 0.92f0
const PASOS_POR_TEMP     = 30

# ─── Cache de medias de fila (n_keep=0, mec_A={}) ─────────────────────────────
# Computed once per node on first use; avoids re-reading 7GB mmap each SA step.
const _rmean_data  = Ref{Union{Nothing, Vector{Float32}}}(nothing)
const _rmean_tpmid = Ref{UInt}(0)

function _get_row_mean(tpm_jl::AbstractMatrix{Float32}, node::Int)::Float32
    tid = objectid(tpm_jl)
    if tid != _rmean_tpmid[]
        _rmean_data[]  = fill(NaN32, size(tpm_jl, 1))
        _rmean_tpmid[] = tid
    end
    cache = _rmean_data[]::Vector{Float32}
    v = cache[node+1]
    if isnan(v)
        v = Float32(mean(view(tpm_jl, node+1, :)))
        cache[node+1] = v
    end
    v
end

# ─── Estado (little-endian) ────────────────────────────────────────────────────
"""
Convierte un vector binario (Int8) a índice de estado little-endian.
estado[1] = bit0 (LSB), estado[n] = bit(n-1) (MSB).
"""
function state_index_le(estado::Vector{Int8})::Int
    s = 0
    for d in 0:length(estado)-1
        s |= (Int(estado[d+1]) << d)
    end
    s
end

# ─── Marginal directo desde TPM ────────────────────────────────────────────────
# Umbral: para n_free > LOOP_MAX usa reducción vectorizada con reshape+mean sin copia extra.
# LOOP_MAX=22: 2^22=4M iters en loop explícito (~0.02s), evita alocar 536 MB (n=27).
const LOOP_MAX = 22

"""
    marginal_directo(tpm_jl, node, keep_dims, estado, n) -> Float32

Calcula E[tpm[node] | keep_dims fijos, resto libre]. Dos ramas:
- n_free ≤ LOOP_MAX: itera solo los 2^n_free estados que coinciden (sin copia).
- n_free > LOOP_MAX: view + reshape sobre mmap (O(1) RAM extra) y media vectorizada.
"""
function marginal_directo(
    tpm_jl::AbstractMatrix{Float32},
    node::Int,
    keep_dims::AbstractVector{Int},
    estado::Vector{Int8},
    n::Int
)::Float32
    n_keep = length(keep_dims)
    n_free = n - n_keep

    # Caso 0: sin dims libres → lectura directa (O(1))
    if n_free == 0
        s = 0
        for d in keep_dims; s |= (Int(estado[d+1]) << d); end
        return tpm_jl[node+1, s+1]
    end

    # Caso 1: sin condicionamiento → media de toda la fila (cacheada)
    if n_keep == 0
        return _get_row_mean(tpm_jl, node)
    end

    if n_free <= LOOP_MAX
        # Loop explícito: itera 2^n_free estados coincidentes (rápido, sin copia)
        keep_set = Set{Int}(keep_dims)
        free_dims = Int[d for d in 0:n-1 if d ∉ keep_set]
        base = 0
        for d in keep_dims; base |= (Int(estado[d+1]) << d); end
        n_states = 1 << n_free
        total = 0f0
        @inbounds for x in 0:n_states-1
            s = base
            for (k, d) in enumerate(free_dims)
                s |= (((x >> (k-1)) & 1) << d)
            end
            total += tpm_jl[node+1, s+1]
        end
        return total * Float32(1.0 / n_states)
    else
        # n_free grande (> LOOP_MAX): usamos view + reshape en lugar de copia
        # para evitar alocar 536 MB (n=27). reshape sobre view es O(1) en RAM.
        row_view = view(tpm_jl, node+1, :)
        tensor = reshape(row_view, ntuple(_ -> 2, n))
        keep_set = Set{Int}(keep_dims)
        free_julia_axes = Tuple(d+1 for d in 0:n-1 if d ∉ keep_set)
        reduced = dropdims(mean(tensor; dims=free_julia_axes); dims=free_julia_axes)
        keep_sorted = sort(collect(keep_set))
        idx = Tuple(Int(estado[d+1]) + 1 for d in keep_sorted)
        return reduced[idx...]
    end
end

# ─── Distribuciones del sistema completo ──────────────────────────────────────
function distribs_sistema(
    tpm_jl::AbstractMatrix{Float32},
    alc_nodes::Vector{Int},
    mec_nodes::Vector{Int},
    estado::Vector{Int8},
    n::Int
)::Vector{Float32}
    [marginal_directo(tpm_jl, node, mec_nodes, estado, n) for node in alc_nodes]
end

# ─── Distribuciones del sistema biparticionado ────────────────────────────────
"""
Para bipartición (alc_A, mec_A):
- Nodos de alcance en alc_A → marginados sobre mec_B = mec_nodes \\ mec_A
- Nodos de alcance en alc_B → marginados sobre mec_A
"""
function distribs_biparticion(
    tpm_jl::AbstractMatrix{Float32},
    alc_nodes::Vector{Int},
    mec_nodes::Vector{Int},
    alc_A::Set{Int},
    mec_A::Set{Int},
    estado::Vector{Int8},
    n::Int
)::Vector{Float32}
    mec_A_list = sort(collect(mec_A))
    mec_B_list = sort([m for m in mec_nodes if m ∉ mec_A])

    result = Vector{Float32}(undef, length(alc_nodes))
    for (i, node) in enumerate(alc_nodes)
        if node ∈ alc_A
            result[i] = marginal_directo(tpm_jl, node, mec_A_list, estado, n)
        else
            result[i] = marginal_directo(tpm_jl, node, mec_B_list, estado, n)
        end
    end
    result
end

# ─── Métrica EMD (L1) ─────────────────────────────────────────────────────────
emd_efecto(u::Vector{Float32}, v::Vector{Float32})::Float32 = sum(abs(ui - vi) for (ui, vi) in zip(u, v))

function phi_biparticion(
    tpm_jl, alc_nodes, mec_nodes, alc_A, mec_A, estado, n, dists_sys
)::Float32
    dp = distribs_biparticion(tpm_jl, alc_nodes, mec_nodes, alc_A, mec_A, estado, n)
    emd_efecto(dp, dists_sys)
end

# ─── Perfiles comprimidos para IB ─────────────────────────────────────────────
"""
En vez de perfiles de 2^26 = 67M elementos, muestrea N_PROFILE_SAMPLES estados
aleatorios. Reduce RAM de ~14 GB a ~200 KB para n=26.
"""
function extraer_perfiles_comprimidos(
    tpm_jl::AbstractMatrix{Float32},
    nodos::Vector{Int};
    n_samples::Int = N_PROFILE_SAMPLES,
    seed::Int = 42
)::Matrix{Float32}
    rng       = MersenneTwister(seed)
    n_states  = size(tpm_jl, 2)
    sample_idxs = rand(rng, 1:n_states, n_samples)

    n_nodos  = length(nodos)
    perfiles = Matrix{Float32}(undef, n_nodos, n_samples)

    for (i, node) in enumerate(nodos)
        for (j, idx) in enumerate(sample_idxs)
            perfiles[i, j] = tpm_jl[node+1, idx]
        end
        s = sum(perfiles[i, :])
        if s > 1f-12
            perfiles[i, :] ./= s
        else
            fill!(view(perfiles, i, :), 1f0 / n_samples)
        end
    end
    perfiles
end

# ─── KL divergencia suavizada ─────────────────────────────────────────────────
function kl_suavizada(p::AbstractVector{Float32}, q::AbstractVector{Float32})::Float32
    total = 0f0
    @inbounds for (pi, qi) in zip(p, q)
        p_ = max(pi, 1f-12)
        q_ = max(qi, 1f-12)
        total += p_ * log(p_ / q_)
    end
    total
end

# ─── Minimización alternada IB ────────────────────────────────────────────────
function ib_alternating(
    perfiles::Matrix{Float32},
    k::Int;
    beta::Float32   = BETA_IB,
    max_iter::Int   = MAX_ITER_IB,
    rng::AbstractRNG = MersenneTwister(42)
)::Vector{Int}
    n, m = size(perfiles)

    pt_x = rand(rng, Float32, n, k)
    pt_x ./= sum(pt_x, dims=2)

    px = fill(1f0 / n, n)

    centroides  = zeros(Float32, k, m)
    log_nuevo   = zeros(Float32, n, k)

    for _ in 1:max_iter
        # p(t) = Σ_i p(t|i) * p(i)
        pt = vec(pt_x' * px)
        pt .= max.(pt, 1f-12)
        pt ./= sum(pt)

        # Centroides: media ponderada de perfiles por cluster
        fill!(centroides, 0f0)
        for t in 1:k
            pesos = pt_x[:, t] .* px
            total = sum(pesos)
            if total > 1f-12
                centroides[t, :] = pesos' * perfiles / total
            else
                centroides[t, :] = vec(mean(perfiles, dims=1))
            end
            centroides[t, :] .= max.(centroides[t, :], 1f-12)
            centroides[t, :] ./= sum(centroides[t, :])
        end

        # Actualizar p(t|i)
        fill!(log_nuevo, 0f0)
        for i in 1:n, t in 1:k
            log_nuevo[i, t] = log(pt[t]) - beta * kl_suavizada(perfiles[i, :], centroides[t, :])
        end
        log_nuevo .-= maximum(log_nuevo, dims=2)
        pt_x_new = exp.(log_nuevo)
        pt_x_new ./= sum(pt_x_new, dims=2)

        if maximum(abs.(pt_x_new .- pt_x)) < 1f-8
            break
        end
        pt_x = pt_x_new
    end

    # Asignación hard (0-indexed)
    [argmax(pt_x[i, :]) - 1 for i in 1:n]
end

# ─── Semilla IB ───────────────────────────────────────────────────────────────
# Mínimo de nodos que deben quedar en cada lado del mecanismo para que
# marginal_directo use el loop explícito (n_free ≤ LOOP_MAX).
# Con n=26 y LOOP_MAX=15, necesitamos |mec_A| ≥ 11 y |mec_B| ≥ 11.
const MEC_MIN_EACH = 2   # mínimo absoluto (permisivo; SA se encargará del resto)

function _asig_a_biparticion(
    asig::Vector{Int},
    nodos::Vector{Int},
    alc_nodes::Vector{Int},
    mec_nodes::Vector{Int};
    rng_fallback::AbstractRNG = MersenneTwister(0)
)::Tuple{Set{Int}, Set{Int}}
    alc_set  = Set(alc_nodes)
    mec_set  = Set(mec_nodes)
    mec_list = sort(collect(mec_set))
    alc_list = sort(collect(alc_set))
    n_mec    = length(mec_list)
    n_alc    = length(alc_list)

    grupo0 = Set(nodos[i] for (i, g) in enumerate(asig) if g == 0)

    mec_A = intersect(grupo0, mec_set)
    alc_A = intersect(grupo0, alc_set)

    # Solo corregir alc_A trivial (alc vacío o alc completo es partición degenerada)
    # mec_A puede ser vacío o completo: la partición (mec_A={}) es válida y a veces óptima
    bad_alc = isempty(alc_A) || alc_A == alc_set
    if bad_alc
        half_a = n_alc ÷ 2
        shuffled_a = shuffle(rng_fallback, alc_list)
        alc_A = Set(shuffled_a[1:half_a])
    end

    alc_A, mec_A
end

function generar_semilla_ib(
    tpm_jl::AbstractMatrix{Float32},
    nodos::Vector{Int},
    alc_nodes::Vector{Int},
    mec_nodes::Vector{Int},
    estado::Vector{Int8},
    n::Int,
    dists_sys::Vector{Float32};
    n_restarts::Int = N_RESTARTS_IB
)::Tuple{Set{Int}, Set{Int}}
    perfiles = extraer_perfiles_comprimidos(tpm_jl, nodos)

    mec_list = sort(collect(mec_nodes))
    alc_list = sort(collect(alc_nodes))

    # Semilla inicial: split equilibrado
    best_alc_A = Set(alc_list[1:length(alc_list)÷2])
    best_mec_A = Set(mec_list[1:length(mec_list)÷2])
    best_perdida = phi_biparticion(tpm_jl, alc_nodes, mec_nodes, best_alc_A, best_mec_A, estado, n, dists_sys)

    # También probar semilla degenerada (mec_A={}): suele ser la MIP óptima en sistemas integrados
    degen_alc_A = Set([first(alc_list)])
    degen_mec_A = Set{Int}()
    degen_perdida = phi_biparticion(tpm_jl, alc_nodes, mec_nodes, degen_alc_A, degen_mec_A, estado, n, dists_sys)
    if degen_perdida < best_perdida
        best_perdida = degen_perdida
        best_alc_A   = degen_alc_A
        best_mec_A   = degen_mec_A
    end

    for r in 0:n_restarts-1
        rng  = MersenneTwister(42 + r)
        asig = ib_alternating(perfiles, 2; rng=rng)
        alc_A, mec_A = _asig_a_biparticion(asig, nodos, alc_nodes, mec_nodes;
                                             rng_fallback=MersenneTwister(100 + r))

        perdida = phi_biparticion(tpm_jl, alc_nodes, mec_nodes, alc_A, mec_A, estado, n, dists_sys)

        if perdida < best_perdida
            best_perdida = perdida
            best_alc_A   = alc_A
            best_mec_A   = mec_A
        end

        perdida <= 1f-12 && break
    end

    best_alc_A, best_mec_A
end

# ─── Simulated Annealing ──────────────────────────────────────────────────────
function sa_biparticion(
    tpm_jl::AbstractMatrix{Float32},
    alc_nodes::Vector{Int},
    mec_nodes::Vector{Int},
    estado::Vector{Int8},
    n::Int,
    dists_sys::Vector{Float32},
    alc_A_init::Set{Int},
    mec_A_init::Set{Int};
    temp_inicial::Float32  = TEMP_INICIAL,
    temp_final::Float32    = TEMP_FINAL,
    factor_enfriam::Float32 = FACTOR_ENFRIAM,
    pasos_por_temp::Int    = PASOS_POR_TEMP,
    seed::Int = 42
)::Tuple{Set{Int}, Set{Int}, Float32}
    alc_nodes_set = Set(alc_nodes)
    mec_nodes_set = Set(mec_nodes)

    alc_A = copy(alc_A_init)
    mec_A = copy(mec_A_init)

    actual_perdida = phi_biparticion(tpm_jl, alc_nodes, mec_nodes, alc_A, mec_A, estado, n, dists_sys)
    mejor_perdida  = actual_perdida
    mejor_alc_A    = copy(alc_A)
    mejor_mec_A    = copy(mec_A)

    mejor_perdida <= 1f-12 && return mejor_alc_A, mejor_mec_A, mejor_perdida

    rng = MersenneTwister(seed + length(alc_nodes))

    # Limitar pasos por nivel: 5 es suficiente con semilla IB de buena calidad
    pasos = 5

    temp = temp_inicial
    niveles_sin_mejora = 0
    mejor_previa       = mejor_perdida

    # Umbral mínimo de mejora para resetear el contador (evita que ruido numérico
    # lo extienda indefinidamente cuando la semilla ya encontró el óptimo).
    min_mejora = 1f-4

    total_verts    = length(mec_nodes) + length(alc_nodes)

    while temp > temp_final
        for _ in 1:pasos
            v_idx = rand(rng, 1:total_verts)
            new_alc_A = copy(alc_A)
            new_mec_A = copy(mec_A)

            if v_idx <= length(mec_nodes)
                node = mec_nodes[v_idx]
                node ∈ mec_A ? delete!(new_mec_A, node) : push!(new_mec_A, node)
            else
                node = alc_nodes[v_idx - length(mec_nodes)]
                node ∈ alc_A ? delete!(new_alc_A, node) : push!(new_alc_A, node)
            end

            # Rechazar biparticiones triviales
            (isempty(new_alc_A) && isempty(new_mec_A)) && continue
            (new_alc_A == alc_nodes_set && new_mec_A == mec_nodes_set) && continue

            nueva_perdida = phi_biparticion(
                tpm_jl, alc_nodes, mec_nodes, new_alc_A, new_mec_A, estado, n, dists_sys
            )
            delta = nueva_perdida - actual_perdida

            if delta < 0 || rand(rng, Float32) < exp(-delta / temp)
                alc_A = new_alc_A
                mec_A = new_mec_A
                actual_perdida = nueva_perdida

                if actual_perdida < mejor_perdida
                    mejor_perdida = actual_perdida
                    mejor_alc_A   = copy(alc_A)
                    mejor_mec_A   = copy(mec_A)
                end
            end
        end

        temp *= factor_enfriam
        mejor_perdida <= 1f-12 && break

        if mejor_perdida < mejor_previa - min_mejora
            mejor_previa        = mejor_perdida
            niveles_sin_mejora  = 0
        else
            niveles_sin_mejora += 1
            if niveles_sin_mejora >= 5 && temp < temp_inicial * 0.5f0
                break
            end
        end
    end

    mejor_alc_A, mejor_mec_A, mejor_perdida
end

# ─── Formateo de partición ─────────────────────────────────────────────────────
function fmt_biparticion(alc_A, mec_A, alc_nodes, mec_nodes)::String
    mec_A_s = sort(collect(mec_A))
    alc_A_s = sort(collect(alc_A))
    mec_B_s = sort([m for m in mec_nodes if m ∉ mec_A])
    alc_B_s = sort([a for a in alc_nodes if a ∉ alc_A])
    "(M=$mec_A_s, A=$alc_A_s) / (M=$mec_B_s, A=$alc_B_s)"
end

# ─── API principal ────────────────────────────────────────────────────────────
"""
    ibqnodos_caso(tpm_jl, alc_str, mec_str, estado_str; warm_alc_A, warm_mec_A)
        -> (phi, particion_str, alc_A, mec_A)

Corre IBQNodos sobre un caso. Retorna phi (Float32), string de partición,
y los sets de la bipartición (para warm-start del siguiente caso).
"""
function ibqnodos_caso(
    tpm_jl::AbstractMatrix{Float32},
    alc_str::String,
    mec_str::String,
    estado_str::String;
    warm_alc_A::Union{Set{Int}, Nothing} = nothing,
    warm_mec_A::Union{Set{Int}, Nothing} = nothing,
    n_total::Int = size(tpm_jl, 1),
    verbose::Bool = false
)::Tuple{Float32, String, Set{Int}, Set{Int}}
    # Parsear strings binarios → listas de nodos 0-indexed
    alc_nodes = [i-1 for (i, c) in enumerate(alc_str) if c == '1']
    mec_nodes = [i-1 for (i, c) in enumerate(mec_str) if c == '1']
    estado    = Int8[parse(Int8, c) for c in estado_str]

    (isempty(alc_nodes) || isempty(mec_nodes)) &&
        return 0f0, "NO-PARTITION", Set{Int}(), Set{Int}()

    n = n_total

    # 1. Distribuciones del sistema completo
    verbose && print("    [IB] distribs_sistema...")
    dists_sys = distribs_sistema(tpm_jl, alc_nodes, mec_nodes, estado, n)
    verbose && (println(" ok"); flush(stdout))

    # 2. Semilla IB
    if warm_alc_A !== nothing && warm_mec_A !== nothing
        # Warm-start: filtrar al conjunto de vértices actual
        # mec_A puede quedar vacío: la partición (mec_A={}) es válida y frecuentemente óptima
        alc_A_init = Set(v for v in warm_alc_A if v ∈ alc_nodes)
        mec_A_init = Set(v for v in warm_mec_A if v ∈ mec_nodes)
        if isempty(alc_A_init)
            alc_A_init = Set([first(alc_nodes)])
        end
        # Evaluar además el candidato degenerado (mec_A={}) y quedarse con el mejor
        alc_list_ws = sort(collect(alc_nodes))
        degen_alc_ws = Set([first(alc_list_ws)])
        degen_mec_ws = Set{Int}()
        phi_ws   = phi_biparticion(tpm_jl, alc_nodes, mec_nodes, alc_A_init, mec_A_init, estado, n, dists_sys)
        phi_degen = phi_biparticion(tpm_jl, alc_nodes, mec_nodes, degen_alc_ws, degen_mec_ws, estado, n, dists_sys)
        if phi_degen < phi_ws
            alc_A_init = degen_alc_ws
            mec_A_init = degen_mec_ws
        end
        verbose && println("    [IB] warm-start aplicado (|alc_A|=$(length(alc_A_init)), |mec_A|=$(length(mec_A_init)))")
    else
        verbose && print("    [IB] generando semilla...")
        nodos = sort(unique(vcat(alc_nodes, mec_nodes)))
        alc_A_init, mec_A_init = generar_semilla_ib(
            tpm_jl, nodos, alc_nodes, mec_nodes, estado, n, dists_sys
        )
        verbose && println(" ok  φ_seed=$(round(phi_biparticion(tpm_jl, alc_nodes, mec_nodes, alc_A_init, mec_A_init, estado, n, dists_sys), digits=4))")
    end

    # 3. SA refinamiento
    verbose && print("    [SA] refinando...")
    best_alc_A, best_mec_A, phi = sa_biparticion(
        tpm_jl, alc_nodes, mec_nodes, estado, n, dists_sys,
        alc_A_init, mec_A_init
    )
    verbose && println(" ok  φ=$(round(phi, digits=6))")

    particion = fmt_biparticion(best_alc_A, best_mec_A, alc_nodes, mec_nodes)
    phi, particion, best_alc_A, best_mec_A
end

end # module
