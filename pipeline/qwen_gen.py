"""Generate consistent-cast scenes with Qwen-Image-Edit-2509 on Kaggle (free GPU).
usage: python pipeline/qwen_gen.py projects/<name>/qshots.json [--variants 1] [--slug bytesized-qwen-edit] [--dtype bf16|fp16]
qshots.json: [{"id":"s01","ref":"byte","prompt":"..."}, {"id":"s02","from":"s01","prompt":"same scene, ..."}]
Cast sheets are cast/<name>_ref.png."""
import json, os, sys, time, shutil, base64, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_scenes import kaggle, OWNER, ROOT

STYLE = ("2D cartoon illustration, clean thin black outlines, soft cel shading, detailed background, warm muted natural colors, "
         "keep each character's design exactly as in the reference: round white head, dot eyes, same clothes and colors")
NEG = "photorealistic, 3d render, text, watermark, extra limbs, deformed hands, blurry, messy lines, gore"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots"); ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--slug", default="bytesized-qwen-edit"); ap.add_argument("--machine", default="NvidiaTeslaT4")
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--no-lightning", action="store_true")
    a = ap.parse_args()
    shots = json.load(open(a.shots))
    cast = {s["ref"] for s in shots if "ref" in s}
    refs = {c: base64.b64encode(open(os.path.join(ROOT, "cast", f"{c}_ref.png"), "rb").read()).decode() for c in cast}
    cfg = dict(style=STYLE, negative=NEG, variants=a.variants, seed=1234, steps=28, true_cfg=4.0, dtype=a.dtype,
               lightning=not a.no_lightning, gguf_repo="QuantStack/Qwen-Image-Edit-2509-GGUF", gguf_quant=a.quant)
    work = f"/tmp/kaggle_push_{a.slug}"
    shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    src = open(os.path.join(ROOT, "kaggle", "qwen_edit_template.py")).read()
    src = src.replace("__SHOTS_JSON__", json.dumps(shots)).replace("__CONFIG_JSON__", json.dumps(cfg)).replace("__REFS_JSON__", json.dumps(refs))
    open(f"{work}/generator.py", "w").write(src)
    json.dump({"id": f"{OWNER}/{a.slug}", "title": a.slug.replace("-", " ").title(), "code_file": "generator.py",
               "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
               "enable_internet": True, "dataset_sources": [], "machine_shape": a.machine},
              open(f"{work}/kernel-metadata.json", "w"), indent=1)
    print(kaggle("kernels", "push", "-p", work))
    ref = f"{OWNER}/{a.slug}"; t0 = time.time()
    while True:
        time.sleep(30)
        st = kaggle("kernels", "status", ref, check=False)
        print(f"[{(time.time()-t0)/60:.1f} min] {st.strip().splitlines()[-1] if st.strip() else ''}", flush=True)
        if any(s in st.lower() for s in ("complete", "error", "cancel")):
            break
    out = os.path.join(os.path.dirname(os.path.abspath(a.shots)), "scenes_qwen")
    os.makedirs(out, exist_ok=True)
    print(kaggle("kernels", "output", ref, "-p", out, check=False)[-1500:])
    print("images:", len([f for f in os.listdir(out) if f.endswith(".png")]), "->", out)

if __name__ == "__main__":
    main()
