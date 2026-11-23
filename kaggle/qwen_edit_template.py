# Kaggle GPU worker: Qwen-Image-Edit-2509 (GGUF 4-bit transformer + nf4 text encoder) with a cast reference image.
# Shots: {"id", "prompt", "ref": "<cast name>"} starts from a cast sheet; {"id", "prompt", "from": "<shot id>"} edits an
# earlier output (pose swap / same scene). Placeholders are filled by pipeline/qwen_gen.py.
import subprocess, sys, os, time, json, base64, io, traceback
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers", "transformers", "accelerate",
                "bitsandbytes", "gguf", "peft"], check=False)
import torch
from PIL import Image

CFG = json.loads(r'''__CONFIG_JSON__''')
SHOTS = json.loads(r'''__SHOTS_JSON__''')
REFS = {k: Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB") for k, v in json.loads(r'''__REFS_JSON__''').items()}
OUT = "/kaggle/working"
NG = torch.cuda.device_count()
print("torch", torch.__version__, "gpus", NG, [torch.cuda.get_device_name(i) for i in range(NG)], flush=True)
DT = torch.bfloat16 if CFG.get("dtype", "bf16") == "bf16" else torch.float16

from huggingface_hub import HfApi, hf_hub_download
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
import diffusers, transformers
print("diffusers", diffusers.__version__, "transformers", transformers.__version__, flush=True)

BASE = "Qwen/Qwen-Image-Edit-2509"
t0 = time.time()
files = HfApi().list_repo_files(CFG["gguf_repo"])
gg = [f for f in files if f.endswith(".gguf") and CFG["gguf_quant"] in f]
print("gguf candidates:", gg, flush=True)
gpath = hf_hub_download(CFG["gguf_repo"], gg[0])
transformer = QwenImageTransformer2DModel.from_single_file(
    gpath, quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT,
    config=BASE, subfolder="transformer")
te_dev = 1 if NG > 1 else 0
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, subfolder="text_encoder", torch_dtype=DT, device_map={"": te_dev},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DT))
pipe = QwenImageEditPlusPipeline.from_pretrained(BASE, transformer=transformer, text_encoder=text_encoder, torch_dtype=DT)
pipe.transformer.to("cuda:0")
EXEC = torch.device(f"cuda:{te_dev}")
pipe.vae.to(EXEC)
def _mv(x, d):
    if torch.is_tensor(x): return x.to(d)
    if isinstance(x, (list, tuple)): return type(x)(_mv(i, d) for i in x)
    if isinstance(x, dict): return {k: _mv(v, d) for k, v in x.items()}
    if hasattr(x, "to_tuple"): return type(x)(*[_mv(i, d) for i in x.to_tuple()])
    return x
if te_dev != 0:   # transformer lives on cuda:0, everything else on cuda:1
    pipe.transformer.register_forward_pre_hook(lambda m, a, k: (_mv(a, "cuda:0"), _mv(k, "cuda:0")), with_kwargs=True)
    pipe.transformer.register_forward_hook(lambda m, a, o: _mv(o, EXEC))
    type(pipe)._execution_device = property(lambda self: EXEC)
print(f"loaded in {time.time()-t0:.0f}s; mem", [f"{torch.cuda.memory_allocated(i)/1e9:.1f}GB" for i in range(NG)], flush=True)

steps, cfg_scale = CFG["steps"], CFG["true_cfg"]
if CFG.get("lightning"):
    try:
        lf = [f for f in HfApi().list_repo_files("lightx2v/Qwen-Image-Lightning")
              if "Edit-2509" in f and "8steps" in f and f.endswith(".safetensors")]
        print("lightning candidates:", lf, flush=True)
        pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name=sorted(lf)[-1])
        steps, cfg_scale = 8, 1.0
        print("lightning LoRA loaded", flush=True)
    except Exception:
        traceback.print_exc(); print("continuing without lightning", flush=True)

done, times = {}, []
for sh in SHOTS:
    if "from" in sh and sh["from"] not in done:
        print(f'skip {sh["id"]}: source {sh["from"]} failed', flush=True); continue
    src = done[sh["from"]] if "from" in sh else REFS[sh["ref"]]
    prompt = f'{sh["prompt"]}. {CFG["style"]}'
    for v in range(CFG.get("variants", 1)):
        t = time.time()
        try:
            img = pipe(image=[src], prompt=prompt, negative_prompt=CFG["negative"], true_cfg_scale=cfg_scale,
                       num_inference_steps=steps, generator=torch.Generator("cpu").manual_seed(CFG["seed"] + v * 7919)).images[0]
        except Exception:
            traceback.print_exc(); continue
        dt = time.time() - t; times.append(dt)
        name = f'{sh["id"]}_{"abcd"[v]}.png'
        img.save(f"{OUT}/{name}")
        px = torch.tensor(list(img.convert("L").getdata()), dtype=torch.float32)
        print(f'{name}: {img.size} {dt:.0f}s mean_lum={px.mean():.0f} std={px.std():.0f}', flush=True)
        if v == 0:
            done[sh["id"]] = img
print(f"images: {len(times)}  avg {sum(times)/max(1,len(times)):.0f}s/image  total {time.time()-t0:.0f}s", flush=True)
