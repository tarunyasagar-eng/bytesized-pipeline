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
print("DONE in", round((time.time() - t0) / 60, 1), "min")
