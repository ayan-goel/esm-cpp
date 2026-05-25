#!/usr/bin/env bash
#
# bench_linux_arm.sh — one-shot esm-cpp benchmark on Linux ARM.
#
# Measures the current default (NEON SDOT INT8) and, if the CPU advertises
# +i8mm, also the opt-in SMMLA path (`ESM_NEON_I8MM=on`). The goal is to
# decide whether Linux ARM needs further optimization work (Phase 15) or
# whether the existing kernel stack already delivers something close to
# the x86/Apple headlines on production cloud ARM hardware.
#
# Tested on GCP C4A (Google Axion, Neoverse V2 — has i8mm), Debian 12.
# Should also work on:
#   - GCP T2A (Ampere Altra, Neoverse N1 — SDOT only, no SMMLA run)
#   - AWS Graviton3/4 (c7g, c8g — both have i8mm)
#   - Any other aarch64 Linux with Debian/Ubuntu + apt
#
# Usage from a fresh aarch64 Debian/Ubuntu instance:
#
#   sudo apt update && sudo apt install -y git
#   git clone https://github.com/<owner>/esm-cpp.git
#   cd esm-cpp
#   bash tools/bench_linux_arm.sh
#
# Idempotent: re-running skips the apt/build/download steps. The
# benchmark JSONs land in benchmarks/results/linux_arm_<host>_*.json
# plus a human-readable markdown summary you can scp back.
#
# Environment variable overrides:
#   MODEL         — esm-cpp shorthand (default: esm2_t33_650M)
#   HF_ID         — full HF id (default: facebook/esm2_t33_650M_UR50D)
#   HOST_TAG      — output filename tag (default: hostname -s)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST_TAG="${HOST_TAG:-$(hostname -s | tr '[:upper:]' '[:lower:]')}"
MODEL="${MODEL:-esm2_t33_650M}"
HF_ID="${HF_ID:-facebook/esm2_t33_650M_UR50D}"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"
mkdir -p "$RESULTS_DIR"

# ----- helpers -----

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }
fail() { printf '\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; }

assert_aarch64() {
    if [[ "$(uname -m)" != "aarch64" ]]; then
        fail "this script is for aarch64 Linux. Got: $(uname -srm)"
        echo "  (Use a GCP C4A, T2A, AWS Graviton, or similar ARM Linux instance.)" >&2
        exit 1
    fi
}

report_hardware() {
    step "hardware"
    if command -v lscpu >/dev/null; then
        lscpu | grep -E "^(Model name|Architecture|Vendor|CPU\(s\)|Thread|BogoMIPS)" \
            | sed 's/^/  /' || true
    fi
    note "kernel: $(uname -srm)"
    note "mem:    $(grep MemTotal /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        note "distro: $(. /etc/os-release && echo "$PRETTY_NAME")"
    fi
    HAS_I8MM=0; HAS_SDOT=0
    if grep -q '\bi8mm\b' /proc/cpuinfo 2>/dev/null; then HAS_I8MM=1; fi
    # Linux exposes SDOT as either 'asimddp' (old kernels) or 'dotprod'
    # (newer). Either is fine.
    if grep -qE '\b(asimddp|dotprod)\b' /proc/cpuinfo 2>/dev/null; then HAS_SDOT=1; fi
    note "features: sdot=$HAS_SDOT  i8mm=$HAS_I8MM"
    export HAS_I8MM HAS_SDOT
}

install_apt_deps() {
    step "apt deps"
    local need=()
    for pkg in build-essential cmake python3-dev python3-venv python3-pip git; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            need+=("$pkg")
        fi
    done
    if [[ ${#need[@]} -eq 0 ]]; then
        note "skip: all apt deps present"
        return
    fi
    note "installing: ${need[*]}"
    sudo apt update -qq
    sudo apt install -y "${need[@]}"
}

ensure_venv() {
    step "venv"
    if [[ ! -d .venv ]]; then
        python3 -m venv .venv
        note "created .venv (Python $(.venv/bin/python --version))"
    else
        note "skip: .venv already exists"
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --quiet --upgrade pip wheel
}

build_esm_cpp() {
    step "esm_cpp install"
    if python -c "import esm_cpp; print(esm_cpp.__version__)" 2>/dev/null; then
        note "skip: esm_cpp already installed ($(python -c 'import esm_cpp; print(esm_cpp.__version__)'))"
        return
    fi
    note "building from source (~5 min on 8 vCPUs)"
    # scikit-build-core needs Release build for sane perf numbers.
    CMAKE_BUILD_TYPE=Release pip install -e . 2>&1 | tail -10
    python -c "
import esm_cpp
print(f'  installed: esm_cpp {esm_cpp.__version__}')
print(f'  host ISA  : {esm_cpp.host_isa()}')
print(f'  current   : {esm_cpp.current_isa()}')
"
}

download_weights() {
    step "weights ($HF_ID)"
    # The bench harness reads from the standard HF cache; snapshot_download
    # places the safetensors there. ~2.5 GB for 650M; ~300 MB for 150M.
    pip install --quiet 'huggingface_hub>=0.20' 'transformers>=4.40' torch \
        || pip install --quiet 'huggingface_hub>=0.20'
    python <<PYEOF
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${HF_ID}',
    allow_patterns=['model.safetensors', 'config.json', 'tokenizer*', '*.txt'])
sz = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fs in os.walk(path) for f in fs
)
print(f'  cached at {path} ({sz/1e9:.1f} GB)')
PYEOF
}

# ----- bench wrappers -----

bench() {
    # Usage: bench <label> <out_basename> <extra esm-cpp-bench args...>
    # Honors an inline env-var prefix (e.g., ESM_NEON_I8MM=on bench ...).
    local label="$1"
    local out_name="$2"
    shift 2
    step "bench: $label"
    local out="$RESULTS_DIR/linux_arm_${HOST_TAG}_${out_name}.json"
    if ! esm-cpp-bench --model "$MODEL" \
            --modes esm-cpp-int8,hf-eager-fp32 \
            --output "$out" "$@"; then
        fail "esm-cpp-bench failed for '$label'"
        return 1
    fi
    python <<PYEOF
import json
d = json.load(open('$out'))
r = d['results']
e = r.get('esm-cpp-int8', {})
h = r.get('hf-eager-fp32', {})
sp = d.get('speedup_int8_vs_hf') or 0.0
print(f"  esm-cpp-int8: p50={e.get('p50_ms', 0):.0f} ms  "
      f"thr={e.get('throughput_seqs_per_s', 0):.2f} seq/s  "
      f"isa={e.get('isa', '?')}")
print(f"  hf-eager:     p50={h.get('p50_ms', 0):.0f} ms  "
      f"thr={h.get('throughput_seqs_per_s', 0):.2f} seq/s")
print(f"  speedup:      {sp:.2f}x HF")
PYEOF
}

# ----- summary -----

write_summary() {
    step "summary"
    local summary="$RESULTS_DIR/linux_arm_${HOST_TAG}_summary.md"
    python <<PYEOF > "$summary"
import glob, json, os
host = "${HOST_TAG}"
model = "${MODEL}"
results_dir = "${RESULTS_DIR}"
files = sorted(glob.glob(f"{results_dir}/linux_arm_{host}_*.json"))

print(f"# Linux ARM bench: {host} — {model}\n")

hw = None
rows = []
for f in files:
    d = json.load(open(f))
    if hw is None:
        hw = d.get('hardware', {})
    label = (os.path.basename(f)
             .replace(f"linux_arm_{host}_", "")
             .replace(".json", "")
             .replace("_", " "))
    cfg = d.get('config', {})
    ds = cfg.get('dataset', '')
    if 'synthetic' in ds:
        shape_label = f"varlen ({cfg.get('num_sequences', '?')} OAS-shape)"
    else:
        shape_label = f"uniform {cfg.get('num_sequences', '?')}x?"
    e = d.get('results', {}).get('esm-cpp-int8', {})
    h = d.get('results', {}).get('hf-eager-fp32', {})
    rows.append({
        'label': label,
        'isa': e.get('isa', '?'),
        'esm_p50': e.get('p50_ms', 0),
        'esm_thr': e.get('throughput_seqs_per_s', 0),
        'hf_p50': h.get('p50_ms', 0),
        'speedup': d.get('speedup_int8_vs_hf') or 0.0,
    })

print(f"**CPU:** {hw.get('cpu_model', '?')} "
      f"({hw.get('cpu_threads', '?')} threads)  ")
print(f"**RAM:** {hw.get('ram_gb', '?')} GB  ")
print(f"**Distro:** {hw.get('platform', '?')}  ")
print(f"**esm.cpp host ISA:** {hw.get('esm_cpp_host_isa', '?')}\n")

print("| Run | ISA | esm-cpp p50 (ms) | esm-cpp seq/s | HF p50 (ms) | Speedup vs HF |")
print("|---|---|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['label']} | {r['isa']} | {r['esm_p50']:.0f} | "
          f"{r['esm_thr']:.2f} | {r['hf_p50']:.0f} | {r['speedup']:.2f}× |")

print()
print("## Decision matrix")
print()
print("Compare against Phase-13 / Phase-11 / Phase-14 headlines:")
print()
print("| Platform | 650M uniform 8×256 vs HF eager FP32 |")
print("|---|---|")
print("| Apple M3 Pro (Phase-14 whole-graph)  | 10.05× |")
print("| Apple M3 Pro (Phase-11 AMX-fp16)     | 2.23×  |")
print("| x86 SPR Xeon (AMX-INT8, default)     | 9.31×  |")
print("| **This host (uniform default)**      | see uniform-default row above |")
print("| **This host (uniform SMMLA opt-in)** | see uniform-smmla row above (if i8mm present) |")
print()
print("Interpretation:")
print("  - >= ~6× HF on uniform: ship Linux ARM as-is, no Phase-15 needed")
print("  - 3-5× HF on uniform: Phase-15 (SMMLA branch-hoist + BF16 attn)")
print("    might lift to 5-7×; decide based on user demand")
print("  - <= ~3× HF on uniform: the SDOT default has a regression; debug")
PYEOF
    cat "$summary"
    echo
    note "wrote $summary"
}

teardown_hint() {
    cat <<'EOF'

=========================================
  All done. To copy results back + clean up:

    # From your laptop:
    gcloud compute scp 'INSTANCE:~/esm-cpp/benchmarks/results/linux_arm_*' . \
        --zone=YOUR_ZONE
    gcloud compute instances delete INSTANCE --zone=YOUR_ZONE
=========================================
EOF
}

# ----- main -----

assert_aarch64
report_hardware
install_apt_deps
ensure_venv
build_esm_cpp
download_weights

# 1. Varlen — the "real scientist workload" (OAS-shaped distribution).
bench "varlen OAS-shape, SDOT default" "650m_varlen_default" \
    --dataset benchmarks/data/synthetic_varlen_v1.fasta \
    --warmup 2 --iters 5

# 2. Uniform 8x256 — matches the M3 / x86 headline shape for direct compare.
bench "uniform 8x256, SDOT default" "650m_uniform_default" \
    --batch 8 --len 256 --warmup 3 --iters 10

# 3. SMMLA opt-in — only on hosts that advertise +i8mm.
if [[ "${HAS_I8MM:-0}" == "1" ]]; then
    ESM_NEON_I8MM=on bench "uniform 8x256, SMMLA opt-in" "650m_uniform_smmla" \
        --batch 8 --len 256 --warmup 3 --iters 10
else
    step "SMMLA opt-in"
    note "skip: CPU does not advertise i8mm in /proc/cpuinfo"
    note "(GCP T2A / AWS Graviton2 / Neoverse N1 lack i8mm. Use C4A / Graviton3+.)"
fi

write_summary
teardown_hint
