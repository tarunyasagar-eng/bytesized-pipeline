# ByteSized AI pipeline: standing rules from the channel owner

- **Stickman characters only.** Every character in every scene must be a stickman with a plain round white head and dot eyes,
  including historical figures, crowds and extras. Hats, wigs, beards and costumes are fine; realistic or human-style faces are not.
  - Enforced in `pipeline/qwen_gen.py` STYLE/NEG.
  - During curation, **reject any image with a human-style face** (e.g. clock_story c041, d057, d058, d062, d063, d064 would fail).
- Recurring cast from `characters.json` (Byte, Nova, Boss, Gramps); keep continuity.
- White captions. Energetic voice. Long-form gets the longer description format with chapters and sources.
- Free services only (no paid APIs or tools) until the channel is monetised. Kaggle GPU budget is about 30 h/week; plan with `plan_qshots.py` (3.2 s holds).
- Remove unclean images: garbled text, watermarks, duplicate Byte in one frame, blank faces, extra limbs.
- **Clean-frame process (owner requirement: every frame clean, no mistakes):**
  1. Write shots with at most 2 main characters, each described differently. Never "three stickmen". Crowds only as tiny distant figures.
     Faces toward the camera "with dot eyes and a small mouth". Byte at most once per frame.
  2. Run `pipeline/qc_frames.py` (flags clones, faceless heads, duplicate Byte). It is a coarse aid with false positives.
  3. Then review EVERY image at large size (its 2x2 review sheets at 768 px), never on tiny contact sheets.
  4. Keep about 20% of the week's GPU for a repair run: regenerate failures with a new seed and a fixed prompt.
     If a frame still fails, swap in a pose-library composite or drop it (hold the previous frame).
- Shorts:
  - Hook card plus impact SFX in the first second, and a meme reaction voice by about 1 s.
  - Reactions on key moments (`react` overlays; CC0 clips in `assets/sfx/voice`).
  - No profanity (no "WTF"), to protect monetisation.
- Sound effects/music: only original synthesis or CC0/clearly licensed for commercial use. No Content-ID evasion.
- Uploads:
  - **Long-form:** always private, with 3 owner checkpoints (topic pick, script approval, final video). The owner publishes, or tells me to schedule.
  - **Shorts:** the owner pre-approved auto-publishing with no review. Schedule **1 Short per day at a different time each day, always at least 12 hours after the previous one** (pick times with `pipeline/short_slots.py`) via `publishAt`:
    upload private with `--publish-at`, or `upload_youtube.py --schedule ID TIME` for an existing upload.
    Nothing was to go public before 2026-12-01 (one-week pause requested on 2026-11-24).
  - Shorts must still follow every rule here (stickman-only, clean images, no profanity, licensed audio).
    If one fails a check, fix it or skip it; never publish a doubtful one.
  - Keep the Shorts queue scheduled at least 7 days ahead. Each weekly long video should yield 5-7 Shorts.
  - Never commit secrets; YouTube credentials live in env vars.
- **Weekly cycle** (Saturday, after the Kaggle reset):
  1. Send 3 topic ideas and the owner picks one.
  2. Script, and the owner approves.
  3. Kaggle images overnight.
  4. Curate, assemble the long video plus Shorts.
  5. Private upload of the long video and the links to the owner. Schedule the Shorts.
