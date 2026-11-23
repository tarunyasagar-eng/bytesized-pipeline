# Kaggle GPU worker: renders shots with SDXL base + Quill Daily LoRA v3.
# The pipeline replaces the __SHOTS_JSON__ / __CONFIG_JSON__ placeholders before pushing.
import os, sys, glob, json, time, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers", "transformers", "accelerate", "safetensors", "peft"], check=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoencoderKL, EulerAncestralDiscreteScheduler

SHOTS = json.loads(r'''__SHOTS_JSON__''')
CFG = json.loads(r'''__CONFIG_JSON__''')
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "| shots:", len(SHOTS))

lora = glob.glob("/kaggle/input/**/quill_daily_lora_v3.safetensors", recursive=True) or glob.glob("/kaggle/input/**/*.safetensors", recursive=True)
assert lora, "LoRA v3 not found in /kaggle/input"
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
                                                 torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lora[0])
pipe.fuse_lora(lora_scale=CFG.get("lora_scale", 0.85))
pipe.vae.enable_tiling()

STYLE = CFG["style_prefix"]
for sh in SHOTS:  # CLIP reads only 77 tokens: warn so the scene never gets truncated
    n = len(pipe.tokenizer(f'{sh["prompt"]}, {STYLE}').input_ids)
    print(f'{sh["id"]}: {n} tokens' + ('  <-- TRUNCATED' if n > 77 else ''))
NEG = CFG["negative"]
W, H = CFG["width"], CFG["height"]
t0 = time.time()
for i, sh in enumerate(SHOTS, 1):
    for v in range(CFG.get("variants", 2)):
        seed = CFG.get("seed_base", 1000) + i * 17 + v * 7919
        g = torch.Generator("cuda").manual_seed(seed)
        full = f'{sh["prompt"]}, {STYLE}'
        img = pipe(prompt=full, prompt_2=full, negative_prompt=NEG, width=W, height=H,
                   num_inference_steps=CFG.get("steps", 40), guidance_scale=CFG.get("guidance", 7.0), generator=g).images[0]
        img.save(f'/kaggle/working/{sh["id"]}_{"abcd"[v]}.png')
        print(f'[{i}/{len(SHOTS)}] {sh["id"]} v{v} seed={seed} {time.time()-t0:.0f}s', flush=True)
print("RENDER DONE in", round((time.time() - t0) / 60, 1), "min")

# ---- automatic quality scoring: CLIP prompt match + sharpness/contrast ----
if CFG.get("score", True):
    import numpy as np
    from transformers import CLIPModel, CLIPProcessor
    del pipe; torch.cuda.empty_cache()
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    scores = {}
    for sh in SHOTS:
        files = sorted(glob.glob(f'/kaggle/working/{sh["id"]}_*.png'))
        if not files: continue
        ims = [Image.open(f).convert("RGB") for f in files]
        inp = proc(text=[sh["prompt"]], images=ims, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            out = clip(**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()})
        sims = (out.logits_per_image[:, 0] / clip.logit_scale.exp()).detach().float().cpu().numpy() * 100
        for f, im, sim in zip(files, ims, sims):
            g = np.asarray(im.convert("L").resize((512, int(512 * im.height / im.width))), dtype=np.float32)
            lap = (g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
            scores[os.path.basename(f)] = dict(clip=round(float(sim), 2), sharp=round(float(lap.var()), 1),
                                               contrast=round(float(g.std()), 1), lum=round(float(g.mean()), 1))
    json.dump(scores, open("/kaggle/working/scores.json", "w"), indent=0)
    print("scored", len(scores), "images")
print("DONE in", round((time.time() - t0) / 60, 1), "min")
