# Fast model plan (SDXL-family LoRA trained on our Qwen frames)

Goal: about 10-25 s/image on Kaggle T4 instead of about 295 s, with **no visible drop in quality**.

## When
After the weekly Kaggle GPU quota resets (Saturday 00:00 UTC). Budget: at most about 4.5 GPU h for training and testing.

## Steps
1. **Licence check**: pick an SDXL-family 2D/anime base whose licence allows monetised YouTube use. Record the licence here.
2. **Dataset**: curated clean Qwen frames from clock_story (and gta6 if on-model).
   - Reject any frame flagged unclean.
   - Caption with the cast names (byte, ...) and the style token `bytesized style`.
   - Aim for about 150-300 images, plus horizontal flips where the character is symmetric.
3. **Train** (Kaggle T4 x2, fp16, kohya/diffusers LoRA): one run, about 3 h, with checkpoints every epoch.
4. **Speed-up**: add a 4-8 step distillation LoRA (e.g. DMD2 / Lightning for SDXL) on top.
5. **Quality gate**: nothing switches unless all of these pass.
   - Re-render 24 shots from qshots_part1.json with the new model, using the same prompts.
   - The side-by-side sheet (Qwen vs new) must show matching:
     - character on-model (face, colours, proportions);
     - clean line art;
     - correct hands and no extra limbs;
     - backgrounds as detailed as Qwen's.
   - Automatic checks: no black or NaN frames, sharpness and contrast at least Qwen's, and a CLIP image similarity check against Byte's reference sheet.
   - Pick the best checkpoint by the sheet, not by the last epoch.
   - **The user reviews the sheet and approves.**
6. **Hybrid rollout** (after approval):
   - The fast model does single-character poses, reactions and backgrounds.
   - Qwen is kept for complex multi-character scenes (about 10-15 per video).
   - If the gate fails, stay on Qwen with the pose-library plan.

## Result: first run (2026-11-28)
- **Licences:** SDXL base 1.0 and SDXL-Lightning are both CreativeML OpenRAIL++-M, so commercial use of outputs is allowed
  (subject to the licence's use restrictions).
- **Data:** 142 clean stickman-only frames; 118 for training, 24 held out for the gate. Rank 32, 2000 steps, 1024x576.
  The whole job took 57 min of GPU.
- **Speed:** 3.7 s/image with Lightning 8-step, 16 s/image at full 28 steps, vs about 295 s/image for Qwen (about 80x faster).
- **Gate sheets:** `docs/fast_model_gate/gate_{0,1,2}.jpg`.
- **Verdict: FAIL as a Qwen replacement.**
  - Style, line art and backgrounds are close to Qwen.
  - Byte is never on-model: cap and hoodie colours swap, and he is duplicated.
  - Some human-style faces (c035, c050 crowd), black-head figures (c003, c011, c029), and weaker prompt adherence
    (wrong character counts and poses).
  - About 60% of non-Byte frames are usable.
- **Decision:** stay on Qwen for characters and build the Byte pose library with Qwen.
- **Next idea to test:** the fast model for **empty backgrounds only** (no characters), which the pose-library compositing needs.
  A Byte-heavy dataset (pose library frames) could fix the character for a second LoRA run later.
