# Pose library plan

Draw the recurring cast once in many poses on transparent backgrounds. Each video then needs mostly
backgrounds and a few one-off scenes; characters are composited on top by code.

## Cast (from characters.json)
| id | look | poses |
|---|---|---|
| byte | orange cap, teal hoodie (narrator) | 48 |
| nova | purple ponytail, round glasses | 32 |
| gramps | grey beard, brown cardigan (history beats) | 28 |
| boss | black suit, red tie, gold watch | 28 |
Total about 136 poses.

## Pose list (byte gets all of these; the others get the subset marked *)
- **Talking / explaining:**
  - neutral talk*
  - pointing left*, pointing right*, pointing up*
  - counting on fingers
  - open-palm shrug*
  - presenting with both hands*
- **Emotions:**
  - happy*, laughing*
  - shocked (hands on cheeks)*
  - scared*, angry*, sad*
  - confused (scratching head)*
  - smug, embarrassed
  - excited jump*
  - facepalm
- **Thinking:**
  - hand on chin*
  - lightbulb idea*
  - looking at the sky
  - reading a book*
  - writing on paper
  - looking through a magnifying glass
- **Action:**
  - walking left*, walking right*
  - running
  - sitting on a chair*, sitting on the ground
  - holding an object (generic, hands ready for a prop)*
  - waving*
  - thumbs up*, thumbs down
  - celebrating with fists up
  - sneaking, falling over
  - carrying a box
  - building with a hammer
  - sleeping
- **Views:** front, 3/4 left and 3/4 right of neutral pose; back view; close-up head (5 expressions).
Each pose has a mouth-open twin for talking flicker on the key 10 talking poses (byte, nova).

## How poses are generated
- Model: the approved fast model if it passes the quality gate; otherwise Qwen-Image-Edit from each
  character's reference sheet (cast/byte_ref.png etc.).
- Prompt: `<character descriptor>, <pose>, full body, plain flat white background, bytesized style`, 1024x1024.
- Cut-out: rembg (free, CPU) to get a transparent PNG with a soft edge. Then trim and save as
  `cast/library/<char>/<pose>.png`.
- `cast/library/poses.json` holds tags (emotion, action, facing, hands_free) used by the planner.
- **Quality review:** a contact sheet per character goes to the user. Off-model, unclean or broken-hand
  poses are regenerated (up to 3 tries) or dropped.

## Compositing (assemble.py, new "comp" shot type)
- Layers: background, then character(s), then props and overlays.
- Placement: left / centre / right thirds, scale by shot type (wide 45%, medium 70%, close 95% of height),
  automatic flip so the character faces the action.
- Integration so it doesn't look pasted on:
  - soft contact shadow under the feet;
  - colour grade the character towards the background's average light (strength about 15%);
  - slight edge blur matched to the background's sharpness;
  - same group camera move over the whole layer stack.
- Life: a pose swap every 1.9-3 s, a 2-3 frame squash/bounce on each swap, a subtle idle breathing scale,
  and mouth-open flicker driven by the voice-over volume.

## Planner (plan_qshots.py)
- For each beat, the planner decides **comp** (library pose + background) or **scene** (full generated
  picture, for interactions, props in hands, crowds, unique moments; at most 10-15 per video).
- Backgrounds are planned once per location and reused across beats (about 15-25 per long video).
- Guest characters: a design sheet plus about 10 poses, added to the library and reused in later videos.
- Costume variants: edit existing poses ("same character, wearing X"); stored as `<char>@<costume>/<pose>.png`.

## Budget
- Library build: about 136 images.
  - Fast model: about 1 GPU h.
  - Qwen: about 11 GPU h (byte first, spread over 1-2 weeks).
- Per long video after: 15-25 backgrounds plus 10-15 scenes plus guests, about 2-6 GPU h (Qwen) or under 1 h (fast).
- Per Short: about 5-8 images.

## Order of work
1. Saturday: fast-model training and quality gate (docs/fast_model_plan.md).
2. Build the byte library (with whichever model passed), then send the contact sheet for review.
3. Build compositing in assemble.py, then do a test edit of one clock chapter as comp shots side by side
   with the original for the user.
4. nova, gramps, boss libraries, then switch the planner to the comp/scene mix after the user approves.
