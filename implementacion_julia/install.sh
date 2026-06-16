#!/usr/bin/env bash
# Instala Julia y las dependencias del proyecto IBQNodos

set -e

echo "=== Instalando Julia ==="
# Descargar Julia 1.10 LTS para Linux x86_64
JULIA_VER="1.10.4"
JULIA_TAR="julia-${JULIA_VER}-linux-x86_64.tar.gz"
JULIA_URL="https://julialang-s3.julialang.org/bin/linux/x64/1.10/${JULIA_TAR}"

if ! command -v julia &>/dev/null; then
    echo "Descargando Julia ${JULIA_VER}..."
    wget -q --show-progress "${JULIA_URL}" -O "/tmp/${JULIA_TAR}"
    tar -xzf "/tmp/${JULIA_TAR}" -C "$HOME"
    echo "export PATH=\"\$HOME/julia-${JULIA_VER}/bin:\$PATH\"" >> "$HOME/.bashrc"
    export PATH="$HOME/julia-${JULIA_VER}/bin:$PATH"
    echo "Julia instalada en $HOME/julia-${JULIA_VER}/bin/"
else
    echo "Julia ya está instalada: $(julia --version)"
fi

echo ""
echo "=== Instalando paquetes Julia ==="
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

julia --project="$PROJ_DIR" -e '
using Pkg
Pkg.instantiate()
Pkg.add(["XLSX", "JSON3"])
println("Paquetes instalados correctamente.")
'

echo ""
echo "=== Verificación ==="
julia --project="$PROJ_DIR" -e '
using XLSX, JSON3, Mmap, Statistics, Random
println("Todas las dependencias OK.")
'

echo ""
echo "=== Listo ==="
echo "Ejecutar con:  julia --project=$PROJ_DIR $PROJ_DIR/main_ibqnodos.jl"
