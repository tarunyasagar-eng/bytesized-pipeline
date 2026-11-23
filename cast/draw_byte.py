"""Byte, ByteSized AI's main character: reference sheet used as the identity image for Qwen-Image-Edit."""
import cairosvg, sys
K = "#1e1a1d"
def S(w=7): return f'stroke="{K}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"'
def byte(x, y, s=1.0, mood="smile", pose="stand"):
    face = {"smile": f'<path d="M-24 22 Q0 44 24 22" {S(6)} fill="none"/>',
            "shock": f'<ellipse cx="0" cy="30" rx="13" ry="17" fill="{K}"/>'}[mood]
    arms = {"stand": f'<path d="M-38 130 L-62 215" {S(10)} fill="none"/><path d="M38 130 L62 215" {S(10)} fill="none"/>',
            "point": f'<path d="M-38 130 L-62 215" {S(10)} fill="none"/><path d="M38 130 L120 95" {S(10)} fill="none"/>'}[pose]
    return f'''<g transform="translate({x},{y}) scale({s})">
  <!-- legs: jeans + sneakers -->
  <path d="M-16 255 L-28 380" stroke="#2d3a5c" stroke-width="20" stroke-linecap="round"/><path d="M16 255 L28 380" stroke="#2d3a5c" stroke-width="20" stroke-linecap="round"/>
  <path d="M-16 255 L-28 380" {S(4)} fill="none" opacity="0"/>
  <path d="M-52 382 Q-50 368 -28 370 L-12 372 Q-6 392 -30 394 L-52 394 Z" fill="#ffffff" {S(5)}/>
  <path d="M52 382 Q50 368 28 370 L12 372 Q6 392 30 394 L52 394 Z" fill="#ffffff" {S(5)}/>
  {arms}
  <path d="M0 78 L0 112" {S(10)}/>
  <!-- teal hoodie torso -->
  <path d="M-46 110 Q0 95 46 110 L52 262 Q0 274 -52 262 Z" fill="#1fb5a8" {S(7)}/>
  <path d="M-30 210 L30 210 L26 245 L-26 245 Z" fill="#17968b" {S(5)}/>
  <path d="M-10 112 L-14 160" {S(4)}/><path d="M10 112 L14 160" {S(4)}/>
  <!-- gloves -->
  {'<circle cx="-62" cy="218" r="13" fill="'+K+'"/>'}{'<circle cx="62" cy="218" r="13" fill="'+K+'"/>' if pose=="stand" else '<circle cx="124" cy="93" r="13" fill="'+K+'"/>'}
  <!-- head -->
  <circle cx="0" cy="0" r="82" fill="#ffffff" {S(8)}/>
  <circle cx="-26" cy="-4" r="9" fill="{K}"/><circle cx="26" cy="-4" r="9" fill="{K}"/>
  <path d="M-42 -28 Q-28 -36 -14 -28" {S(6)} fill="none"/><path d="M14 -28 Q28 -36 42 -28" {S(6)} fill="none"/>
  {face}
  <!-- orange cap -->
  <path d="M-80 -22 Q-78 -92 0 -94 Q78 -92 80 -22 Z" fill="#ff8c1a" {S(7)}/>
  <path d="M30 -30 Q95 -42 128 -20 Q100 -6 60 -16 Z" fill="#e06f00" {S(6)}/>
  <circle cx="0" cy="-92" r="7" fill="#e06f00" {S(4)}/>
</g>'''
W, H = 1024, 576
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" fill="#ffffff"/>
{byte(250, 120, 1.0, "smile", "stand")}{byte(560, 150, 0.92, "smile", "point")}{byte(860, 150, 0.92, "shock", "stand")}
</svg>'''
cairosvg.svg2png(bytestring=svg.encode(), write_to=sys.argv[1], output_width=W, output_height=H)
print("wrote", sys.argv[1])
