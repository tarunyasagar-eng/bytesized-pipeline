"""Plain ByteSized stickman (no clothes): identity base for costumed historical characters."""
import cairosvg, sys
K = "#1e1a1d"
def S(w=7): return f'stroke="{K}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"'
def stick(x, y, s, mood, pose):
    mouth = {"smile": f'<path d="M-24 22 Q0 44 24 22" {S(6)} fill="none"/>', "think": f'<path d="M-16 28 L18 24" {S(6)}/>',
             "shock": f'<ellipse cx="0" cy="30" rx="13" ry="17" fill="{K}"/>'}[mood]
    arms = {"stand": "M0 150 L-55 250 M0 150 L55 250", "point": "M0 150 L-55 250 M0 150 L110 110",
            "chin": "M0 150 L-55 250 M0 150 L40 200 L18 70"}[pose]
    return f'''<g transform="translate({x},{y}) scale({s})">
 <path d="M0 80 L0 270 M0 270 L-45 400 M0 270 L45 400 {arms}" {S(11)} fill="none"/>
 <circle cx="0" cy="0" r="82" fill="#fff" {S(8)}/>
 <circle cx="-26" cy="-4" r="9" fill="{K}"/><circle cx="26" cy="-4" r="9" fill="{K}"/>
 <path d="M-42 -28 Q-28 -36 -14 -28" {S(6)} fill="none"/><path d="M14 -28 Q28 -36 42 -28" {S(6)} fill="none"/>{mouth}</g>'''
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="576"><rect width="1024" height="576" fill="#fff"/>
{stick(240,110,0.98,"smile","stand")}{stick(540,130,0.92,"think","chin")}{stick(830,130,0.92,"shock","point")}</svg>'''
cairosvg.svg2png(bytestring=svg.encode(), write_to=sys.argv[1], output_width=1024, output_height=576)
print("wrote", sys.argv[1])
