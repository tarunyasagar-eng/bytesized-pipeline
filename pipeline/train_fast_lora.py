"""Train the fast ByteSized SDXL LoRA on Kaggle (T4) and render the quality-gate set. See docs/fast_model_plan.md.
Needs the private Kaggle dataset tarunyasagar/bytesized-train-clock (images + captions.json).
usage: python pipeline/train_fast_lora.py evalset.json [--steps 2000] [--attach]"""
import argparse, json, os, shutil, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OWNER = "tarunyasagar"
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from qwen_gen import NEG
def kaggle(*args, check=True):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "kaggle_cli.py"), *args], capture_output=True, text=True)
    return r.stdout + r.stderr
ap = argparse.ArgumentParser(); ap.add_argument("evalset"); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--slug", default="bytesized-fast-lora"); ap.add_argument("--dataset", default=f"{OWNER}/bytesized-train-clock")
ap.add_argument("--out", default=os.path.join(ROOT, "projects", "fast_lora")); ap.add_argument("--attach", action="store_true")
a = ap.parse_args()
cfg = dict(height=576, width=1024, rank=32, lr=1e-4, steps=a.steps, ckpts=[int(a.steps * 0.4), int(a.steps * 0.7), a.steps],
           train_budget_s=int(2.6 * 3600), negative=NEG)
work = f"/tmp/kaggle_push_{a.slug}"; shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
src = open(os.path.join(ROOT, "kaggle", "sdxl_lora_template.py")).read()
src = src.replace("__CONFIG_JSON__", json.dumps(cfg)).replace("__EVAL_JSON__", json.dumps(json.load(open(a.evalset))))
open(f"{work}/trainer.py", "w").write(src)
json.dump({"id": f"{OWNER}/{a.slug}", "title": a.slug.replace("-", " ").title(), "code_file": "trainer.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": [a.dataset], "machine_shape": "NvidiaTeslaT4"}, open(f"{work}/kernel-metadata.json", "w"), indent=1)
pr = "" if a.attach else kaggle("kernels", "push", "-p", work); print(pr[-600:])
if "push error" in pr.lower(): raise SystemExit("push rejected: " + pr.strip().splitlines()[-1])
ref = f"{OWNER}/{a.slug}"; t0 = time.time()
while True:
    time.sleep(60)
    st = kaggle("kernels", "status", ref, check=False)
    print(f"[{(time.time()-t0)/60:.1f} min] {st.strip().splitlines()[-1] if st.strip() else ''}", flush=True)
    if any(s in st.lower() for s in ("complete", "error", "cancel")): break
os.makedirs(a.out, exist_ok=True)
print(kaggle("kernels", "output", ref, "-p", a.out, check=False)[-1500:])
