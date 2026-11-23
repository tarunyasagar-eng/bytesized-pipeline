"""Push a shot list to Kaggle, run LoRA v3 on GPU, download the images.
usage: python pipeline/gen_scenes.py projects/<name>/shots.json [--format short|long]"""
import json, os, subprocess, sys, time, shutil, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KCLI = [sys.executable, os.path.join(ROOT, "tools", "kaggle_cli.py")]
OWNER = os.environ.get("KAGGLE_USERNAME", "tarunyasagar")

STYLE = "quill_daily style, 2d vector stickman, round white head, dot eyes, clean black outlines, flat cel colors"
NEG = ("photorealistic human, realistic person, human skin, flesh tone, realistic face, photo, realistic eyes, realistic lips, 3d, "
       "realistic anatomy, detailed face, flesh, realistic skin, ugly, blurry, text, watermark, subtitles, words, letters, labels, "
       "bad anatomy, cropped head, extra limbs, deformed, messy lines, sketch, noisy, jpeg artifacts, gore")
FORMATS = {"short": dict(width=768, height=1344, tag="vertical"),
           "long": dict(width=1344, height=768, tag="widescreen"),
           "thumb": dict(width=1344, height=768, tag="widescreen, bold simple composition")}

def kaggle(*args, check=True):
    r = subprocess.run(KCLI + list(args), capture_output=True, text=True, cwd="/tmp")
    if check and r.returncode:
        raise SystemExit(r.stdout + r.stderr)
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots"); ap.add_argument("--format", default="short"); ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--slug", default="bytesized-scene-generator")
    ap.add_argument("--machine", default="NvidiaTeslaT4")
    ap.add_argument("--only", help="comma-separated shot ids to render (for quick tests)")
    a = ap.parse_args()
    shots = json.load(open(a.shots))
    if a.only:
        shots = [x for x in shots if x["id"] in a.only.split(",")]
    fmt = FORMATS[a.format]
    cfg = dict(style_prefix=f"{STYLE}, {fmt['tag']}", negative=NEG, width=fmt["width"], height=fmt["height"],
               variants=a.variants, steps=40, guidance=7.0, lora_scale=0.85)
    work = f"/tmp/kaggle_push_{a.slug}"
    shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    src = open(os.path.join(ROOT, "kaggle", "scene_generator_template.py")).read()
    src = src.replace("__SHOTS_JSON__", json.dumps(shots)).replace("__CONFIG_JSON__", json.dumps(cfg))
    open(f"{work}/generator.py", "w").write(src)
    json.dump({"id": f"{OWNER}/{a.slug}", "title": a.slug.replace("-", " ").title(), "code_file": "generator.py",
               "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
               "enable_internet": True, "dataset_sources": [f"{OWNER}/quill-daily-lora-v3-model"],
               "machine_shape": a.machine}, open(f"{work}/kernel-metadata.json", "w"), indent=1)
    print(kaggle("kernels", "push", "-p", work))
    ref = f"{OWNER}/{a.slug}"
    t0 = time.time()
    while True:
        time.sleep(30)
        st = kaggle("kernels", "status", ref, check=False)
        print(f"[{(time.time()-t0)/60:.1f} min] {st.strip()}", flush=True)
        if any(s in st.lower() for s in ("complete", "error", "cancel")):
            break
    out = os.path.join(os.path.dirname(os.path.abspath(a.shots)), f"scenes_{a.format}")
    os.makedirs(out, exist_ok=True)
    print(kaggle("kernels", "output", ref, "-p", out, check=False)[-2000:])
    print("images:", len([f for f in os.listdir(out) if f.endswith(".png")]), "->", out)

if __name__ == "__main__":
    main()
