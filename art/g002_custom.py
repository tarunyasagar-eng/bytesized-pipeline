"""Hand-drawn replacement for GTA6 shot g002 (2026 living room, same stickman, older with grey beard)."""
import cairosvg, sys
W, H = 1344, 768
K = "#1e1a1d"
def stroke(w=6): return f'stroke="{K}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"'
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
 <linearGradient id="wall" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3b1a6e"/><stop offset="1" stop-color="#7a1f7a"/></linearGradient>
 <linearGradient id="floor" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#4a1f5c"/><stop offset="1" stop-color="#2a1238"/></linearGradient>
 <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff4fa3"/><stop offset="0.55" stop-color="#ff9a3c"/><stop offset="1" stop-color="#2a1a5e"/></linearGradient>
 <radialGradient id="glow" cx="0.5" cy="0.4" r="0.6"><stop offset="0" stop-color="#ff6ad5" stop-opacity="0.55"/><stop offset="1" stop-color="#ff6ad5" stop-opacity="0"/></radialGradient>
 <linearGradient id="win" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1a1046"/><stop offset="1" stop-color="#5a2a8a"/></linearGradient>
</defs>
<rect width="{W}" height="{H}" fill="url(#wall)"/>
<!-- neon strip along ceiling -->
<rect x="0" y="40" width="{W}" height="10" fill="#36f1cd"/><rect x="0" y="32" width="{W}" height="26" fill="#36f1cd" opacity="0.18"/>
<!-- window with night city -->
<rect x="70" y="120" width="250" height="300" rx="8" fill="url(#win)" {stroke(7)}/>
<g fill="#2b1650" {stroke(3)}>
 <rect x="90" y="270" width="45" height="150"/><rect x="140" y="220" width="55" height="200"/><rect x="200" y="300" width="40" height="120"/><rect x="245" y="250" width="60" height="170"/>
</g>
<g fill="#ffd166"><rect x="152" y="240" width="10" height="10"/><rect x="172" y="270" width="10" height="10"/><rect x="258" y="275" width="10" height="10"/><rect x="282" y="310" width="10" height="10"/><rect x="100" y="300" width="8" height="8"/><rect x="210" y="330" width="8" height="8"/></g>
<circle cx="270" cy="170" r="22" fill="#fff4c2" {stroke(3)}/>
<line x1="195" y1="120" x2="195" y2="420" {stroke(6)}/><line x1="70" y1="270" x2="320" y2="270" stroke="{K}" stroke-width="0"/>
<!-- huge TV with tropical neon city -->
<rect x="430" y="95" width="560" height="320" rx="14" fill="{K}"/>
<rect x="448" y="112" width="524" height="286" rx="6" fill="url(#sky)"/>
<circle cx="710" cy="260" r="70" fill="#ffe066"/>
<g fill="#ffe066" opacity="0.9"><rect x="630" y="285" width="160" height="6"/><rect x="645" y="300" width="130" height="5"/><rect x="665" y="313" width="90" height="4"/></g>
<g fill="#2a1a5e"><rect x="470" y="250" width="40" height="148"/><rect x="515" y="210" width="30" height="188"/><rect x="550" y="270" width="45" height="128"/><rect x="830" y="230" width="38" height="168"/><rect x="872" y="190" width="30" height="208"/><rect x="906" y="260" width="50" height="138"/></g>
<rect x="448" y="340" width="524" height="58" fill="#1b2a6b"/>
<g {stroke(5)} fill="none"><path d="M600 398 Q605 330 590 290"/><path d="M800 398 Q795 330 812 288"/></g>
<g fill="#1e6b4a" {stroke(4)}><path d="M590 290 q-40 -5 -60 20 q35 -10 60 -20z"/><path d="M590 290 q30 -30 65 -15 q-35 0 -65 15z"/><path d="M590 290 q-10 -35 -45 -45 q30 25 45 45z"/>
<path d="M812 288 q40 -5 60 20 q-35 -10 -60 -20z"/><path d="M812 288 q-30 -30 -65 -15 q35 0 65 15z"/><path d="M812 288 q10 -35 45 -45 q-30 25 -45 45z"/></g>
<rect x="430" y="95" width="560" height="320" rx="14" fill="none" {stroke(7)}/>
<rect x="400" y="80" width="620" height="350" rx="30" fill="url(#glow)"/>
<!-- tv stand -->
<rect x="470" y="430" width="480" height="60" rx="6" fill="#2e1846" {stroke(6)}/>
<rect x="560" y="445" width="90" height="28" rx="4" fill="#111" {stroke(4)}/><circle cx="635" cy="459" r="5" fill="#36f1cd"/>
<rect x="770" y="445" width="110" height="28" rx="4" fill="#111" {stroke(4)}/><rect x="780" y="455" width="40" height="5" fill="#ff4fa3"/>
<!-- calendar 2026 -->
<rect x="1070" y="110" width="170" height="190" rx="8" fill="#fff" {stroke(6)}/>
<rect x="1070" y="110" width="170" height="52" rx="8" fill="#e63946" {stroke(6)}/>
<circle cx="1110" cy="110" r="7" fill="#fff" {stroke(4)}/><circle cx="1200" cy="110" r="7" fill="#fff" {stroke(4)}/>
<text x="1155" y="250" font-family="Lilita One" font-size="70" text-anchor="middle" fill="{K}">2026</text>
<!-- floor -->
<path d="M0 520 L{W} 520 L{W} {H} L0 {H} Z" fill="url(#floor)"/>
<line x1="0" y1="520" x2="{W}" y2="520" {stroke(6)}/>
<ellipse cx="760" cy="660" rx="420" ry="70" fill="#ff4fa3" opacity="0.25"/>
<ellipse cx="760" cy="660" rx="420" ry="70" fill="none" stroke="#ff4fa3" stroke-width="5"/>
<!-- plant -->
<rect x="1190" y="440" width="70" height="80" rx="6" fill="#c46a2d" {stroke(6)}/>
<g fill="#2fa36b" {stroke(5)}><path d="M1225 440 q-60 -40 -50 -120 q40 50 50 120z"/><path d="M1225 440 q60 -50 45 -130 q-40 60 -45 130z"/><path d="M1225 440 q0 -80 10 -150 q20 80 -10 150z"/></g>
<!-- couch -->
<rect x="440" y="520" width="560" height="120" rx="26" fill="#3a2a8c" {stroke(7)}/>
<rect x="470" y="470" width="500" height="90" rx="22" fill="#4b3ab0" {stroke(7)}/>
<rect x="410" y="500" width="80" height="150" rx="26" fill="#3a2a8c" {stroke(7)}/>
<rect x="950" y="500" width="80" height="150" rx="26" fill="#3a2a8c" {stroke(7)}/>
<line x1="720" y1="475" x2="720" y2="560" {stroke(5)}/>
<!-- older stickman sitting on couch, facing viewer, holding controller -->
<g {stroke(9)} fill="none">
 <path d="M720 455 L720 560"/>
 <path d="M720 560 L650 575 L640 680"/><path d="M720 560 L790 575 L800 680"/>
 <path d="M622 682 L655 682"/><path d="M785 682 L820 682"/>
 <path d="M720 485 L670 535 L700 560"/><path d="M720 485 L770 535 L740 560"/>
</g>
<!-- controller -->
<rect x="680" y="545" width="80" height="34" rx="16" fill="#222" {stroke(6)}/>
<circle cx="745" cy="556" r="5" fill="#ff4fa3"/><circle cx="745" cy="569" r="5" fill="#36f1cd"/>
<rect x="690" y="558" width="16" height="5" fill="#fff"/><rect x="695" y="553" width="5" height="15" fill="#fff"/>
<!-- head -->
<circle cx="720" cy="385" r="78" fill="#ffffff" {stroke(8)}/>
<!-- grey beard -->
<path d="M652 405 Q660 500 720 505 Q780 500 788 405 Q760 440 720 440 Q680 440 652 405 Z" fill="#b9b9c4" {stroke(6)}/>
<path d="M700 452 Q720 465 740 452" {stroke(5)} fill="none"/>
<!-- smile line in beard -->
<path d="M698 428 Q720 446 742 428" {stroke(6)} fill="none"/>
<!-- eyes + bushy grey brows -->
<circle cx="693" cy="375" r="9" fill="{K}"/><circle cx="747" cy="375" r="9" fill="{K}"/>
<path d="M672 350 Q692 338 710 350" stroke="#9a9aa6" stroke-width="10" stroke-linecap="round" fill="none"/>
<path d="M730 350 Q748 338 768 350" stroke="#9a9aa6" stroke-width="10" stroke-linecap="round" fill="none"/>
<!-- wrinkles -->
<path d="M700 330 Q720 324 740 330" stroke="{K}" stroke-width="3" fill="none"/>
<!-- sparkle of joy -->
<g fill="#ffe066" {stroke(3)}><path d="M830 320 l8 20 l20 8 l-20 8 l-8 20 l-8 -20 l-20 -8 l20 -8z"/><path d="M600 300 l5 13 l13 5 l-13 5 l-5 13 l-5 -13 l-13 -5 l13 -5z"/></g>
</svg>'''
out = sys.argv[1]
cairosvg.svg2png(bytestring=svg.encode(), write_to=out, output_width=W, output_height=H)
print("wrote", out)
