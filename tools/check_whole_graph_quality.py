"""Phase 13 T7: logit-quality + small-scope PPPL drift for the whole-graph
ANE/GPU path. Per CLAUDE.md the strict gates are:

  PPPL  drift |whole_graph - fp32| < 0.1
  ProteinGym Spearman drift < 0.01

This script measures the cheaper logit-corr + argmax-agreement proxy across
the 25-protein holdout (mirroring tools/check_logit_quality.py's pattern),
then runs a small-scope (3-5 sequences) PPPL drift sweep so the strict gate
gets at least one concrete number per model size before ship.

Usage:
    .venv-arm/bin/python tools/check_whole_graph_quality.py \\
        --model esm2_t30_150M --num-seqs 8 --pppl-num 3

The script auto-builds a (B=1, L) whole-graph artifact for each unique L in
the holdout (cached to /tmp/wg_quality_<model>/L-<L>/...).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import esm_cpp


HF_IDS = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
}

# Same 25-protein holdout as tools/run_pppl_drift.py (kept in sync).
HOLDOUT = [
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKR",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "MASKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
    "GDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
    "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSI",
    "SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
    "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "MTKQEKTALNMARFIRSQTLTLLEKLNELDADEQADICESLHDHADELYRSCLARFGDDGENL",
    "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN",
    "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK",
    "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA",
    "MTTASTSQVRQNYHQDSEAAINRQINLELYASYVYLSMSYYFDRDDVALKNFAKYFLHQSHEEREHAEKLMKLQNQRGGRIFLQDIKKPDCDDWESGLNAMECALHLEKNVNQSLLELHKLATDKNDPHLCDFIETHYLNEQVKAIKELGDHVTNLRKMGAPESGLAEYLFDKHTLGDSDNES",
    "MGLSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSKHPGDFGADAQGAMTKALELFRNDIAAKYKELGFQG",
    "GIVEQCCTSICSLYQLENYCN",
    "MVKQIESKTAFQEALDAAGDKLVVVDFSATWCGPCKMIKPFFHSLSEKYSNVIFLEVDVDDCQDVASECEVKCMPTFQFFKKGQKVGEFSGANKEKLEATINELV",
    "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV",
    "KIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFN",
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQH",
    "SAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS",
    "MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNRVAQQLAGKQSLLIGVATSSLALHAPSQIVAAIKSRADQLGASVVVSMVERSGVEACKAAVHNLLAQRVSGLIINYPLDDQDAIAVEAACTNVPALFLDVSDQTPINSII",
    "KETAAAKFERQHMDSSTSAA",
    "TEFKAGSAKKGATLFKTRCLQCHTVEKGGPHKVGPNLHGIFGRHSGQAEGYSYTDANIKKNVLWDENNMSEYLTNPKKYIPGTKMAFGGLKKEKDRNDLITYLKKACE",
]


def _safetensors_path(hf_id: str) -> Path:
    cache = Path.home() / ".cache/huggingface/hub" / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    cands = list(cache.glob("*/model.safetensors"))
    if not cands:
        sys.exit(f"HF cache miss for {hf_id}")
    return cands[0]


def _build_artifact(hf_id: str, B: int, L: int, out_root: Path,
                    ct_python: Path) -> Path:
    """Build a (B, L) whole-graph .mlmodelc if not already cached."""
    tag = f"B-{B}_L-{L}"
    target_dir = out_root / tag
    bundle = target_dir / "whole_graph.mlmodelc"
    if bundle.exists():
        return bundle
    print(f"  [build] {hf_id} @ {tag}")
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ct_python), "tools/build_whole_graph_artifacts.py",
        "--model", hf_id, "--shapes", f"{B}x{L}", "--out", str(out_root),
        "--precision", "fp16", "--compute-units", "CPU_AND_NE",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit("artifact build failed")
    if not bundle.exists():
        raise SystemExit(f"artifact missing after build: {bundle}")
    return bundle


def _logit_quality(model: esm_cpp.Model, tok: esm_cpp.Tokenizer,
                   sequences: list[str], out_root: Path, hf_id: str,
                   ct_python: Path) -> dict:
    """corr + argmax agreement between FP32 forward and whole-graph forward.

    For each sequence, builds (B=1, L) whole-graph artifact if not cached,
    measures both FP32 (no env, no artifacts loaded) and whole-graph
    (env=on, artifact registered) on the SAME model instance.
    """
    rows = []
    os.environ.pop("ESM_APPLE_AMX", None)
    os.environ.pop("ESM_APPLE_ANE_GRAPH", None)

    fp32_logits_by_idx = []
    print("[fp32] computing reference logits")
    for i, seq in enumerate(sequences):
        ids = np.asarray(tok.encode(seq), dtype=np.int32)
        logits = model.forward(ids)
        if not np.isfinite(logits).all():
            raise RuntimeError(f"FP32 logits non-finite for seq {i}")
        fp32_logits_by_idx.append(logits)
        print(f"  [{i:2d}] L={ids.size}  fp32 ok")

    # Build artifacts + measure whole-graph (env on for each).
    print("[whole-graph] building artifacts + measuring")
    os.environ["ESM_APPLE_ANE_GRAPH"] = "on"
    for i, seq in enumerate(sequences):
        ids = np.asarray(tok.encode(seq), dtype=np.int32)
        L = int(ids.size)
        bundle = _build_artifact(hf_id, 1, L, out_root, ct_python)
        ok = model.load_whole_graph_artifact(str(bundle), 1, L, "cpu_and_ne")
        if not ok:
            raise RuntimeError(f"failed to register whole-graph artifact for L={L}")
        out = model.forward_scheduled([ids])
        wg_logits = out[0]
        ref = fp32_logits_by_idx[i]
        if wg_logits.shape != ref.shape:
            raise RuntimeError(f"shape mismatch: wg {wg_logits.shape} vs fp32 {ref.shape}")
        corr = float(np.corrcoef(ref.ravel(), wg_logits.ravel())[0, 1])
        agree = float((ref.argmax(-1) == wg_logits.argmax(-1)).mean())
        finite = bool(np.isfinite(wg_logits).all())
        rows.append({"idx": i, "L": L, "corr": corr, "argmax_agree": agree,
                     "finite": finite})
        print(f"  [{i:2d}] L={L:3d}  corr={corr:.6f}  argmax={agree:.4f}  finite={finite}")

    corrs = [r["corr"] for r in rows]
    agrees = [r["argmax_agree"] for r in rows]
    return {
        "rows": rows,
        "min_corr": min(corrs),
        "median_corr": float(np.median(corrs)),
        "min_argmax_agree": min(agrees),
        "median_argmax_agree": float(np.median(agrees)),
        "any_nan": any(not r["finite"] for r in rows),
    }


def _pppl_one_path(model: esm_cpp.Model, tok: esm_cpp.Tokenizer,
                   sequence: str, mask_id: int) -> tuple[float, int]:
    """Per-sequence PPPL summand (sum NLL, count). Uses model.forward() to
    keep each forward at (B=1, L) — engages the whole-graph path when env+
    registration are set up. Returns the running sum / position count."""
    ids = np.asarray(tok.encode(sequence), dtype=np.int32)
    L = int(ids.size)
    nll_sum = 0.0
    cnt = 0
    for p in range(1, L - 1):
        variant = ids.copy()
        variant[p] = mask_id
        # Use forward() to get a [L, V] logits matrix at one masked position.
        # Whole-graph dispatch lives in forward_scheduled, so call that and
        # take the first (and only) output.
        out = model.forward_scheduled([variant])
        logits_row = out[0][p]
        m = float(logits_row.max())
        log_p = float(logits_row[ids[p]] - m - np.log(np.exp(logits_row - m).sum()))
        nll_sum += -log_p
        cnt += 1
    return nll_sum, cnt


def _pppl_drift(model: esm_cpp.Model, tok: esm_cpp.Tokenizer,
                sequences: list[str], out_root: Path, hf_id: str,
                ct_python: Path) -> dict:
    """Compute PPPL_fp32 vs PPPL_whole_graph on a small subset (gate < 0.1)."""
    mask_id = esm_cpp.Tokenizer.mask_id

    # PPPL fp32 reference
    os.environ.pop("ESM_APPLE_AMX", None)
    os.environ.pop("ESM_APPLE_ANE_GRAPH", None)
    print("[pppl fp32] computing reference")
    t = time.perf_counter()
    sum_nll, cnt = 0.0, 0
    for i, s in enumerate(sequences):
        s_nll, s_cnt = _pppl_one_path(model, tok, s, mask_id)
        sum_nll += s_nll; cnt += s_cnt
        print(f"  [{i}] L={len(s)+2}  positions={s_cnt}")
    fp32_pppl = float(np.exp(sum_nll / cnt))
    fp32_elapsed = time.perf_counter() - t
    print(f"  PPPL_fp32 = {fp32_pppl:.6f}  ({fp32_elapsed:.1f} s)")

    # PPPL whole-graph
    print("[pppl whole-graph] computing")
    os.environ["ESM_APPLE_ANE_GRAPH"] = "on"
    # Pre-register one (B=1, L) artifact per unique L in this subset
    for s in sequences:
        L = len(tok.encode(s))
        bundle = _build_artifact(hf_id, 1, L, out_root, ct_python)
        ok = model.load_whole_graph_artifact(str(bundle), 1, L, "cpu_and_ne")
        if not ok:
            raise RuntimeError(f"register failed for L={L}")
    t = time.perf_counter()
    sum_nll, cnt = 0.0, 0
    for i, s in enumerate(sequences):
        s_nll, s_cnt = _pppl_one_path(model, tok, s, mask_id)
        sum_nll += s_nll; cnt += s_cnt
        print(f"  [{i}] L={len(s)+2}  positions={s_cnt}")
    wg_pppl = float(np.exp(sum_nll / cnt))
    wg_elapsed = time.perf_counter() - t
    print(f"  PPPL_wg = {wg_pppl:.6f}  ({wg_elapsed:.1f} s)")

    drift = abs(wg_pppl - fp32_pppl)
    return {
        "fp32_pppl": fp32_pppl,
        "whole_graph_pppl": wg_pppl,
        "drift": drift,
        "fp32_elapsed_s": fp32_elapsed,
        "wg_elapsed_s": wg_elapsed,
        "num_sequences": len(sequences),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF_IDS))
    ap.add_argument("--num-seqs", type=int, default=8,
                    help="logit-quality holdout subset size")
    ap.add_argument("--pppl-num", type=int, default=3,
                    help="PPPL drift subset size (smaller = faster)")
    ap.add_argument("--ct-python", default="/tmp/ct312/bin/python")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    hf_id = HF_IDS[args.model]
    sft = _safetensors_path(hf_id)
    out_root = Path(f"/tmp/wg_quality_{args.model}")

    print(f"[load] {hf_id}")
    model = esm_cpp.Model.load_from_safetensors(str(sft))
    tok = esm_cpp.Tokenizer()

    seqs_q = HOLDOUT[: args.num_seqs]
    seqs_pppl = HOLDOUT[: args.pppl_num]
    print(f"holdout: {len(seqs_q)} for logit quality, {len(seqs_pppl)} for PPPL")

    print("\n===== logit quality =====")
    q = _logit_quality(model, tok, seqs_q, out_root, hf_id, Path(args.ct_python))

    print("\n===== PPPL drift =====")
    p = _pppl_drift(model, tok, seqs_pppl, out_root, hf_id, Path(args.ct_python))

    summary = {
        "model": args.model,
        "hf_id": hf_id,
        "num_quality": len(seqs_q),
        "num_pppl": len(seqs_pppl),
        "logit_quality": q,
        "pppl_drift": p,
        "gates": {
            "logit_corr_min_pass": q["min_corr"] >= 0.999,
            "argmax_min_pass": q["min_argmax_agree"] >= 0.99,
            "pppl_drift_pass": p["drift"] < 0.1,
        }
    }
    print("\n===== summary =====")
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
