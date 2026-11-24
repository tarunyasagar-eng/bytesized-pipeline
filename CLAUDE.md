# ByteSized AI pipeline: standing rules from the channel owner

- **Stickman characters only.** Every character in every scene must be a stickman with a plain round white head and dot eyes,
  including historical figures, crowds and extras. Hats, wigs, beards and costumes are fine; realistic or human-style faces are not.
  - Enforced in `pipeline/qwen_gen.py` STYLE/NEG.
  - During curation, **reject any image with a human-style face** (e.g. clock_story c041, d057, d058, d062, d063, d064 would fail).
- Recurring cast from `characters.json` (Byte, Nova, Boss, Gramps); keep continuity.
- White captions. Energetic voice. Long-form gets the longer description format with chapters and sources.
- Free services only (no paid APIs or tools) until the channel is monetised. Kaggle GPU budget is about 30 h/week; plan with `plan_qshots.py` (3.2 s holds).
- Remove unclean images: garbled text, watermarks, duplicate Byte in one frame, blank faces, extra limbs.
- Shorts:
  - Hook card plus impact SFX in the first second, and a meme reaction voice by about 1 s.
  - Reactions on key moments (`react` overlays; CC0 clips in `assets/sfx/voice`).
  - No profanity (no "WTF"), to protect monetisation.
- Sound effects/music: only original synthesis or CC0/clearly licensed for commercial use. No Content-ID evasion.
- Uploads: always private (or scheduled), for the owner's review. Never commit secrets; YouTube credentials live in env vars.
