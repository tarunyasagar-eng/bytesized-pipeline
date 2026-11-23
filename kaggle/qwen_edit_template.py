# Kaggle GPU worker: Qwen-Image-Edit-2509 (GGUF 4-bit transformer + nf4 text encoder) with a cast reference image.
# Shots: {"id", "prompt", "ref": "<cast name>"} starts from a cast sheet; {"id", "prompt", "from": "<shot id>"} edits an
# earlier output (pose swap / same scene). Placeholders are filled by pipeline/qwen_gen.py.
import subprocess, sys, os, time, json, base64, io, traceback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers", "transformers", "accelerate",
                "bitsandbytes", "gguf", "peft"], check=False)
import torch
import torch.nn.functional as F
from PIL import Image

# T4 has no bf16 flash/mem-efficient SDPA, so attention falls back to the math path and materializes the full
# heads x L x L matrix. Slice the queries: exact same result, peak memory / (L / QCHUNK).
_sdpa, QCHUNK = F.scaled_dot_product_attention, 1024
def _sliced_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    L = query.shape[-2]
    if L <= QCHUNK or is_causal:
        return _sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    outs = []
    for i in range(0, L, QCHUNK):
        m = attn_mask
        if m is not None and m.ndim >= 2 and m.shape[-2] == L:
            m = m[..., i:i + QCHUNK, :]
        outs.append(_sdpa(query[..., i:i + QCHUNK, :], key, value, attn_mask=m, dropout_p=dropout_p, scale=scale, **kw))
    return torch.cat(outs, dim=-2)
F.scaled_dot_product_attention = _sliced_sdpa

CFG = json.loads(r'''__CONFIG_JSON__''')
SHOTS = json.loads(r'''__SHOTS_JSON__''')
REFS = {k: Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB") for k, v in json.loads(r'''__REFS_JSON__''').items()}
OUT = "/kaggle/working"
NG = torch.cuda.device_count()
print("torch", torch.__version__, "gpus", NG, [torch.cuda.get_device_name(i) for i in range(NG)], flush=True)
# T4 runs fp16 natively (~5x faster than emulated bf16) but Qwen-Image overflows in fp16 -> black frames.
# "mixed": transformer in fp16 with per-block clamping (as ComfyUI does); text encoder + VAE stay bf16.
DT = torch.bfloat16 if CFG.get("dtype", "bf16") in ("bf16", "mixed") else torch.float16
TDT = torch.float16 if CFG.get("dtype") in ("fp16", "mixed") else DT

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
    gpath, quantization_config=GGUFQuantizationConfig(compute_dtype=TDT), torch_dtype=TDT,
    config=BASE, subfolder="transformer")
te_dev = 1 if NG > 1 else 0
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, subfolder="text_encoder", torch_dtype=DT, device_map={"": te_dev},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DT))
pipe = QwenImageEditPlusPipeline.from_pretrained(BASE, transformer=transformer, text_encoder=text_encoder, torch_dtype=DT)
pipe.transformer.to("cuda:0")
EXEC = torch.device(f"cuda:{te_dev}")
pipe.vae.to(EXEC)
pipe.vae.enable_tiling()   # VAE encode/decode of a ~1MP image OOMs on a T4 without tiling
def _mv(x, d, dt=None):
    if torch.is_tensor(x):
        x = x.to(d)
        if dt is not None and x.is_floating_point():
            if dt == torch.float16 and x.dtype != torch.float16:
                x = x.float().clamp(-65000, 65000)   # bf16 activations can exceed the fp16 range
            x = x.to(dt)
        return x
    if isinstance(x, (list, tuple)): return type(x)(_mv(i, d, dt) for i in x)
    if isinstance(x, dict): return {k: _mv(v, d, dt) for k, v in x.items()}
    if hasattr(x, "to_tuple"): return type(x)(*[_mv(i, d, dt) for i in x.to_tuple()])
    return x
# transformer lives on cuda:0 in TDT; everything else on EXEC in DT
pipe.transformer.register_forward_pre_hook(lambda m, a, k: (_mv(a, "cuda:0", TDT), _mv(k, "cuda:0", TDT)), with_kwargs=True)
NANS = [0]
def _post(m, a, o):
    t = o[0] if isinstance(o, (tuple, list)) else getattr(o, "sample", o)
    if torch.is_tensor(t) and not torch.isfinite(t).all(): NANS[0] += 1
    return _mv(o, EXEC, DT)
pipe.transformer.register_forward_hook(_post)
if te_dev != 0:
    type(pipe)._execution_device = property(lambda self: EXEC)
if TDT == torch.float16:
    # Output projections (attn to_out / to_add_out, FFN down-proj) overflow fp16: their inputs reach ~1e4 and are
    # summed over thousands of dims. Feed them input/S and return output*S in fp32 (same math); the gated residual
    # add runs in fp32 and the residual stream stays fp32. Every linear casts its (normalized) input to fp16.
    FS = CFG.get("ffn_scale", 32.0)
    def _to16(x):
        if torch.is_tensor(x) and x.is_floating_point() and x.dtype != torch.float16:
            x = x.float().clamp(-65000, 65000).half()
        return x
    def _pre(scale):
        return lambda m, a: (_to16(a[0] / scale if scale != 1 else a[0]),) + tuple(a[1:])
    def _post_scaled(m, a, o):
        if not torch.isfinite(o).all():   # rare (large late-layer weights): redo this one call 32x further down-scaled
            o = type(m).forward(m, (a[0].float() / 32).half(), *a[1:]).float() * 32
        return o.float() * FS
    SCALED = ("attn.to_out.0", "attn.to_add_out", "img_mlp.net.2", "txt_mlp.net.2")
    nsc = 0
    for b in pipe.transformer.transformer_blocks:
        # residual stream stays fp32 (no _clamp): Qwen's late-layer activations exceed the fp16 range for real
        for n, mod in b.named_modules():
            if not isinstance(mod, torch.nn.Linear) and "Linear" not in type(mod).__name__:
                continue
            if any(n == k or n.startswith(k + ".base_layer") for k in SCALED):
                mod.register_forward_pre_hook(_pre(FS)); mod.register_forward_hook(_post_scaled); nsc += 1
            else:
                mod.register_forward_pre_hook(_pre(1))
    for mod in (pipe.transformer.proj_out, pipe.transformer.norm_out.linear):
        mod.register_forward_pre_hook(_pre(1))
    print("fp16 scaled projections:", nsc, flush=True)
    if CFG.get("debug_nan"):   # report the first submodule whose output goes non-finite, plus input ranges
        FIRST = []
        def _dbg(name):
            def h(m, a, o):
                if FIRST: return
                t = o[0] if isinstance(o, (tuple, list)) else o
                if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                    ins = [f"{x.dtype}:{x.float().abs().max().item():.3g}" for x in a if torch.is_tensor(x) and x.is_floating_point()]
                    FIRST.append(name); print(f"FIRST NONFINITE: {name} ({type(m).__name__}) inputs absmax {ins}", flush=True)
            return h
        for n, mod in pipe.transformer.named_modules():
            if n: mod.register_forward_hook(_dbg(n))
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

# T4 has no bf16 flash/mem-efficient attention, so attention is materialized: keep token count down and
# spread the transformer over both GPUs (last SPLIT blocks on cuda:1).
import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as QM
QM.VAE_IMAGE_SIZE = CFG.get("vae_area", 640 * 640)
if NG > 1:
    blocks = pipe.transformer.transformer_blocks
    for b in list(blocks)[len(blocks) - CFG.get("split", 14):]:
        b.to("cuda:1")
    for b in blocks:
        dev = next(b.parameters()).device
        b.register_forward_pre_hook(lambda m, a, k, d=dev: (_mv(a, d), _mv(k, d)), with_kwargs=True)
    pipe.transformer.norm_out.register_forward_pre_hook(lambda m, a: _mv(a, "cuda:0"))
    torch.cuda.empty_cache()
print("after split mem", [f"{torch.cuda.memory_allocated(i)/1e9:.1f}GB" for i in range(NG)], flush=True)

done, times = {}, []
for sh in SHOTS:
    if "from" in sh and sh["from"] not in done:
        print(f'skip {sh["id"]}: source {sh["from"]} failed', flush=True); continue
    src = done[sh["from"]] if "from" in sh else REFS[sh["ref"]]
    prompt = f'{sh["prompt"]}. {CFG["style"]}'
    for v in range(CFG.get("variants", 1)):
        t = time.time()
        try:
            for i in range(NG): torch.cuda.empty_cache()
            NANS[0] = 0
            img = pipe(image=[src], prompt=prompt, height=CFG.get("height", 576), width=CFG.get("width", 1024), negative_prompt=CFG["negative"], true_cfg_scale=cfg_scale,
                       num_inference_steps=steps, generator=torch.Generator("cpu").manual_seed(CFG["seed"] + v * 7919)).images[0]
        except Exception:
            traceback.print_exc(); print("mem", [f"{torch.cuda.memory_allocated(i)/1e9:.1f}GB" for i in range(NG)], flush=True); continue
        dt = time.time() - t; times.append(dt)
        name = f'{sh["id"]}_{"abcd"[v]}.png'
        img.save(f"{OUT}/{name}")
        px = torch.tensor(list(img.convert("L").getdata()), dtype=torch.float32)
        print(f'{name}: {img.size} {dt:.0f}s nonfinite_steps={NANS[0]} mean_lum={px.mean():.0f} std={px.std():.0f}', flush=True)
        if v == 0:
            done[sh["id"]] = img
print(f"images: {len(times)}  avg {sum(times)/max(1,len(times)):.0f}s/image  total {time.time()-t0:.0f}s", flush=True)
