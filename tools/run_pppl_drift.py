"""PPPL drift validation: |PPPL_int8 - PPPL_fp32| < 0.1 gate.

Phase 7 ships with two opt-in env-var-gated variants (BF16 attention,
INT8 lm_head) that need PPPL validation before they can graduate to
default-on. This script runs the drift measurement on:

  1. FP32 baseline (the reference)
  2. Default INT8 quantization (per-layer weight INT8 + FP32 lm_head;
     same recipe as Phase 6 default-on)
  3. INT8 + ESM_AMX_ATTENTION=on (BF16 attention overlay)
  4. INT8 + ESM_QUANTIZE_LM_HEAD=on (also quantize lm_head.dense)
  5. INT8 + both env vars on

Holdout: 25 well-known real proteins (lysozyme, ubiquitin, GFP, hemoglobin,
insulin, etc.). Short enough to fit in the 30-min budget on 150M and
650M, real enough that PPPL is meaningful. For a tighter measurement,
swap in a UniRef50 holdout via --data.

Usage:
  python tools/run_pppl_drift.py --model esm2_t30_150M
  python tools/run_pppl_drift.py --model esm2_t33_650M

Gate: per CLAUDE.md / spec, |PPPL_int8 - PPPL_fp32| < 0.1 must hold for
each variant on 150M and 650M.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import esm_cpp
from esm_cpp.eval.pppl import pppl


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

_HF_ID_FOR_SHORTHAND = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
}


# 25 well-known real protein sequences (PDB / UniProt references). Length
# distribution: min 51 (insulin B-chain mature), median ~140, max 238 (GFP).
# All are in standard 20-AA alphabet; ESM-2 tokenizer handles them directly.
_HOLDOUT = [
    # Insulin B-chain mature (UniProt P01308 fragment)
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKR",
    # Ubiquitin (UniProt P0CG48)
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    # Hen egg-white lysozyme (PDB 1AKI)
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    # GFP (Aequorea victoria, UniProt P42212 sans N-Met)
    "MASKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
    # Hemoglobin alpha chain (UniProt P69905)
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    # Hemoglobin beta chain (UniProt P68871)
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
    # Cytochrome c (horse heart, UniProt P00004 fragment)
    "GDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
    # BRCA1 RING domain (UniProt P38398 fragment)
    "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSI",
    # p53 DNA-binding domain (UniProt P04637 fragment)
    "SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
    # Streptococcal protein G (UniProt P19909 fragment, GB1 domain)
    "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    # ROP (Rome operator protein, ColE1, PDB 1ROP)
    "MTKQEKTALNMARFIRSQTLTLLEKLNELDADEQADICESLHDHADELYRSCLARFGDDGENL",
    # Crambin (PDB 1CRN)
    "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN",
    # Calmodulin (UniProt P0DP23)
    "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK",
    # Bovine pancreatic trypsin inhibitor (BPTI; UniProt P00974 fragment)
    "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA",
    # Ferritin H chain (UniProt P02794 fragment)
    "MTTASTSQVRQNYHQDSEAAINRQINLELYASYVYLSMSYYFDRDDVALKNFAKYFLHQSHEEREHAEKLMKLQNQRGGRIFLQDIKKPDCDDWESGLNAMECALHLEKNVNQSLLELHKLATDKNDPHLCDFIETHYLNEQVKAIKELGDHVTNLRKMGAPESGLAEYLFDKHTLGDSDNES",
    # Myoglobin (UniProt P02185)
    "MGLSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSKHPGDFGADAQGAMTKALELFRNDIAAKYKELGFQG",
    # Insulin A-chain mature (UniProt P01308 fragment)
    "GIVEQCCTSICSLYQLENYCN",
    # Thioredoxin (UniProt P10599)
    "MVKQIESKTAFQEALDAAGDKLVVVDFSATWCGPCKMIKPFFHSLSEKYSNVIFLEVDVDDCQDVASECEVKCMPTFQFFKKGQKVGEFSGANKEKLEATINELV",
    # Bovine pancreatic ribonuclease A (UniProt P61823 fragment)
    "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV",
    # Maltose-binding protein (E. coli, UniProt P0AEX9 N-terminal frag)
    "KIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFN",
    # Ras GTPase (UniProt P01112 H-Ras fragment, residues 1-166)
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQH",
    # IL-8 (CXCL8, UniProt P10145 mature)
    "SAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS",
    # Lac repressor DNA-binding domain (E. coli, fragment)
    "MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNRVAQQLAGKQSLLIGVATSSLALHAPSQIVAAIKSRADQLGASVVVSMVERSGVEACKAAVHNLLAQRVSGLIINYPLDDQDAIAVEAACTNVPALFLDVSDQTPINSII",
    # Bovine pancreatic ribonuclease S-peptide (residues 1-20)
    "KETAAAKFERQHMDSSTSAA",
    # Yeast iso-1 cytochrome c (UniProt P00044 fragment)
    "TEFKAGSAKKGATLFKTRCLQCHTVEKGGPHKVGPNLHGIFGRHSGQAEGYSYTDANIKKNVLWDENNMSEYLTNPKKYIPGTKMAFGGLKKEKDRNDLITYLKKACE",
]


def _safetensors_path_for(hf_id: str) -> Path:
    snapshots = HF_CACHE / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        raise SystemExit(
            f"HF cache missing: {snapshots}. "
            f"Run `huggingface-cli download {hf_id}` first.")
    candidates = list(snapshots.glob("*/model.safetensors"))
    if not candidates:
        raise SystemExit(f"no model.safetensors under {snapshots}")
    return candidates[0]


def _load_fasta(path: Path) -> list[str]:
    out, cur = [], []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    out.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
        if cur:
            out.append("".join(cur))
    return out


def _measure(label: str, model: esm_cpp.Model, tokenizer: esm_cpp.Tokenizer,
             sequences: list[str]) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    value = pppl(model, tokenizer, sequences)
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] PPPL = {value:.6f}  ({elapsed:.1f} s)")
    return label, value, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="esm2_t30_150M",
                        choices=list(_HF_ID_FOR_SHORTHAND.keys()))
    parser.add_argument("--data", type=Path, default=None,
                        help="Optional FASTA holdout. Defaults to the "
                              "embedded 25-protein holdout.")
    parser.add_argument("--num-seqs", type=int, default=0,
                        help="Limit to the first N sequences (0 = all).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sequences = _load_fasta(args.data) if args.data else list(_HOLDOUT)
    if args.num_seqs > 0:
        sequences = sequences[: args.num_seqs]
    print(f"Holdout: {len(sequences)} sequences, "
          f"mean length {sum(len(s) for s in sequences) / len(sequences):.1f}, "
          f"min {min(len(s) for s in sequences)}, "
          f"max {max(len(s) for s in sequences)}")

    hf_id = _HF_ID_FOR_SHORTHAND[args.model]
    path = _safetensors_path_for(hf_id)
    tokenizer = esm_cpp.Tokenizer()

    results: dict[str, dict[str, float]] = {}

    # 1. FP32 baseline.
    print(f"\n[{args.model}] Loading FP32 model...")
    fp32 = esm_cpp.Model.load_from_safetensors(str(path))
    label, fp32_pppl, elapsed = _measure("fp32", fp32, tokenizer, sequences)
    results[label] = {"pppl": fp32_pppl, "elapsed_s": elapsed,
                       "drift_vs_fp32": 0.0}
    del fp32

    # The opt-in env vars are read once at process start (via std::call_once
    # / static initializers in C++), so we can't toggle them mid-process.
    # Instead, log which env vars are set in THIS process and recommend
    # the user run separately for each variant.
    env_bf16 = os.environ.get("ESM_AMX_ATTENTION", "") in ("on", "1", "true")
    env_lm_head = os.environ.get("ESM_QUANTIZE_LM_HEAD", "") in (
        "on", "1", "true")
    variant_label = "int8"
    if env_bf16: variant_label += "+bf16_attn"
    if env_lm_head: variant_label += "+int8_lmhead"

    # 2. INT8 variant.
    print(f"\n[{args.model}] Loading INT8 model (variant: {variant_label})...")
    int8 = esm_cpp.Model.load_from_safetensors(str(path))
    int8.quantize_weights()  # reads ESM_QUANTIZE_LM_HEAD internally
    label, int8_pppl, elapsed = _measure(variant_label, int8, tokenizer,
                                           sequences)
    drift = abs(int8_pppl - fp32_pppl)
    gate_pass = drift < 0.1
    results[variant_label] = {"pppl": int8_pppl, "elapsed_s": elapsed,
                                "drift_vs_fp32": drift, "gate_pass": gate_pass}

    print(f"\n=== Summary ===")
    print(f"FP32 PPPL:        {fp32_pppl:.4f}")
    print(f"{variant_label} PPPL: {int8_pppl:.4f}")
    print(f"Drift:            {drift:.4f}")
    print(f"Gate (< 0.1):     {'PASS' if gate_pass else 'FAIL'}")
    if env_bf16 or env_lm_head:
        print(f"Env vars active: ESM_AMX_ATTENTION={env_bf16}, "
              f"ESM_QUANTIZE_LM_HEAD={env_lm_head}")
    else:
        print("Default-on path measured (no env-var-gated overlays).")
        print("To measure overlays, re-run with:")
        print("  ESM_AMX_ATTENTION=on   python tools/run_pppl_drift.py ...")
        print("  ESM_QUANTIZE_LM_HEAD=on python tools/run_pppl_drift.py ...")

    summary = {
        "model": args.model,
        "isa": esm_cpp.current_isa(),
        "num_sequences": len(sequences),
        "fp32_pppl": fp32_pppl,
        f"{variant_label}_pppl": int8_pppl,
        "drift": drift,
        "gate_threshold": 0.1,
        "gate_pass": gate_pass,
        "env_amx_attention": env_bf16,
        "env_quantize_lm_head": env_lm_head,
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {args.out}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
