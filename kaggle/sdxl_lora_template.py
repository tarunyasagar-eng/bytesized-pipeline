# Kaggle GPU worker: train a ByteSized style/cast LoRA on SDXL base (fp16, T4) from our curated Qwen frames, then render
# the held-out quality-gate prompts with each checkpoint (+ SDXL-Lightning 8-step) and once at full 28 steps.
# Placeholders are filled by pipeline/train_fast_lora.py. Licences: SDXL base 1.0 and SDXL-Lightning are CreativeML OpenRAIL++-M.
import subprocess, sys, os, json, time, glob, math, random, traceback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers", "transformers", "accelerate", "peft",
                "bitsandbytes"], check=False)
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image

CFG = json.loads(r'''__CONFIG_JSON__''')
EVAL = json.loads(r'''__EVAL_JSON__''')
OUT = "/kaggle/working"; T0 = time.time()
def log(*a): print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
cap_files = glob.glob("/kaggle/input/**/captions.json", recursive=True)
DS = os.path.dirname(cap_files[0]); CAPS = json.load(open(cap_files[0]))
log("dataset", DS, len(CAPS), "gpus", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler
BASE = "stabilityai/stable-diffusion-xl-base-1.0"
dev = torch.device("cuda:0"); H, W = CFG["height"], CFG["width"]
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(BASE, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(dev)
log("loaded SDXL")

# ---------- 1. pre-encode latents + text (then free the encoders) ----------
data = []
with torch.no_grad():
    empty = pipe.encode_prompt("", device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
    empty = (empty[0].cpu(), empty[2].cpu())
    for sid, cap in CAPS.items():
        im = Image.open(os.path.join(DS, sid + ".jpg")).convert("RGB").resize((W, H), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].to(dev, torch.float16) / 127.5 - 1
        lat = pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
        pe, _, pooled, _ = pipe.encode_prompt(cap, device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
        data.append((lat.cpu(), pe.cpu(), pooled.cpu()))
log("encoded", len(data))
unet = pipe.unet
del pipe.text_encoder, pipe.text_encoder_2, pipe.vae
torch.cuda.empty_cache()

# ---------- 2. LoRA training ----------
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
import bitsandbytes as bnb
unet.requires_grad_(False)
unet.enable_gradient_checkpointing()
unet.add_adapter(LoraConfig(r=CFG["rank"], lora_alpha=CFG["rank"], init_lora_weights="gaussian",
                            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
params = [p for p in unet.parameters() if p.requires_grad]
for p in params: p.data = p.data.float()
log("lora params", sum(p.numel() for p in params) / 1e6, "M")
opt = bnb.optim.AdamW8bit(params, lr=CFG["lr"], weight_decay=1e-2)
steps, warm = CFG["steps"], 100
sched_lr = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
noise_sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")
acp = noise_sched.alphas_cumprod.to(dev)
scaler = torch.cuda.amp.GradScaler()
tid = torch.tensor([[H, W, 0, 0, H, W]], device=dev, dtype=torch.float16)
def save(step):
    d = f"{OUT}/lora_{step}"
    StableDiffusionXLPipeline.save_lora_weights(d, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)))
    log("saved", d)
unet.train(); run = 0.0; done = []
for step in range(1, steps + 1):
    lat, pe, pooled = random.choice(data)
    if random.random() < 0.08: pe, pooled = empty
    lat = lat.to(dev).float(); pe = pe.to(dev); pooled = pooled.to(dev)
    noise = torch.randn_like(lat) + 0.03 * torch.randn(lat.shape[0], lat.shape[1], 1, 1, device=dev)   # small offset noise
    t = torch.randint(0, 1000, (1,), device=dev)
    noisy = noise_sched.add_noise(lat, noise, t)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = unet(noisy.half(), t, encoder_hidden_states=pe, added_cond_kwargs={"text_embeds": pooled, "time_ids": tid}).sample
    snr = acp[t] / (1 - acp[t]); wgt = torch.clamp(snr, max=5.0) / snr                       # min-SNR-5 weighting (eps pred)
    loss = (F.mse_loss(pred.float(), noise, reduction="none").mean(dim=[1, 2, 3]) * wgt).mean()
    scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched_lr.step()
    run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 50 == 0: log(f"step {step}/{steps} loss {run:.4f} lr {sched_lr.get_last_lr()[0]:.2e}")
    if step in CFG["ckpts"]: save(step); done.append(step)
    if time.time() - T0 > CFG["train_budget_s"]:
        log("training time budget reached"); save(step); done.append(step); break
del unet, opt, data; pipe = None; import gc; gc.collect(); torch.cuda.empty_cache()

# ---------- 3. quality-gate renders ----------
from huggingface_hub import hf_hub_download
light = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_lora.safetensors")
pipe = StableDiffusionXLPipeline.from_pretrained(BASE, vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
                                                 torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(dev)
base_sched = pipe.scheduler.config
NEG = CFG["negative"]; times = {}
def render(tag, light_on, steps_n, cfg):
    tt = []
    for sid, prompt in EVAL.items():
        g = torch.Generator(dev).manual_seed(1234)
        t1 = time.time()
        kw = dict(prompt=prompt, height=H, width=W, num_inference_steps=steps_n, generator=g, guidance_scale=cfg)
        if cfg > 1: kw["negative_prompt"] = NEG
        img = pipe(**kw).images[0]
        tt.append(time.time() - t1); img.save(f"{OUT}/{tag}_{sid}.png")
    times[tag] = round(sum(tt[1:]) / max(1, len(tt) - 1), 2); log(tag, "avg s/image", times[tag])
for ck in done:
    try:
        pipe.unload_lora_weights()
        pipe.load_lora_weights(f"{OUT}/lora_{ck}", adapter_name="bs")
        pipe.load_lora_weights(light, adapter_name="light"); pipe.set_adapters(["bs", "light"], [1.0, 1.0])
        pipe.scheduler = EulerDiscreteScheduler.from_config(base_sched, timestep_spacing="trailing")
        render(f"L8_{ck}", True, 8, 1.0)
    except Exception:
        traceback.print_exc()
try:   # full-quality reference with the last checkpoint (no speed-up)
    ck = done[-1]; pipe.unload_lora_weights(); pipe.load_lora_weights(f"{OUT}/lora_{ck}", adapter_name="bs")
    pipe.scheduler = EulerDiscreteScheduler.from_config(base_sched); render(f"F28_{ck}", False, 28, 6.0)
except Exception:
    traceback.print_exc()
json.dump({"times": times, "ckpts": done}, open(f"{OUT}/report.json", "w"))
log("done", times)
