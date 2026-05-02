import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

C_DISSEM  = '#4C72B0'
C_EXTRACT = '#DD8452'
C_BIDIR   = '#55A868'
C_DEFENSE = '#8172B2'
C_MAS     = '#C44E52'
C_SA      = '#937860'
C_MASNAT  = '#7B5EA7'
C_LLM     = '#5D8A5E'
C_INVERT  = '#e74c3c'
C_MONO    = '#27ae60'

FAMILY_COLOR = {'Dissem.': C_DISSEM, 'Extract.': C_EXTRACT,
                'Bidir.': C_BIDIR, 'Defense': C_DEFENSE}
ORIGIN_COLOR = {'MAS': C_MAS, 'MAS*': C_MASNAT, 'SA': C_SA, 'LLM': C_LLM}
ORIGIN_LABEL = {'MAS': 'General MAS', 'MAS*': 'MAS-native',
                'SA': 'Single-agent LLM', 'LLM': 'LLM RecSys'}

attacks = [
    ('NetSafe',       'Dissem.',  'MAS',  'Both',      'User↑',  'Item↑',  'R'),
    ('CheatAgent',    'Dissem.',  'SA',   'User mem.',  'User↑',  'Item↑',  'C'),
    ('DrunkAgent',    'Dissem.',  'MAS*', 'Item mem.',  'User↑',  'Item↑',  'R'),
    ('PI / CORBA',    'Dissem.',  'MAS',  'Item mem.',  'User↑',  'Item↑',  'C'),
    ('RecTextAttack', 'Dissem.',  'LLM',  'Token',      'Both↑',  'Both↑',  'C'),
    ('MAMA',          'Extract.', 'MAS',  'Both',       'Both↑',  'Both↑',  'C'),
    ('InjecAgent',    'Extract.', 'MAS',  'Item mem.',  'Both↑',  'Both↑',  'C'),
    ('MASLeak',       'Extract.', 'MAS',  'Item mem.',  'Both↑',  'Both↑',  'C'),
    ('TOMA',          'Bidir.',   'MAS',  'Item mem.',  'Both↑',  'Item↑',  'R'),
    ('MASTER',        'Bidir.',   'MAS',  'User mem.',  'Both↑',  'Stable', 'C'),
    ('G-Safeguard',   'Defense',  'MAS',  '',           '',       '',       ''),
    ('BlindGuard',    'Defense',  'MAS',  '',           '',       '',       ''),
    ('T-Guard',       'Defense',  'MAS',  '',           '',       '',       ''),
    ('M-Guard',       'Defense',  'MAS',  '',           '',       '',       ''),
]

ENTRY_COLOR = {'Both': '#55A868', 'User mem.': C_DISSEM,
               'Item mem.': C_EXTRACT, 'Token': '#937860', '': None}

# ── FIGURE 1: taxonomy grid ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

n = len(attacks)
L, R, TOP, BOT = 0.01, 0.99, 0.90, 0.03
# cols: Method | Origin | Entry Point | k_inf effect | rho effect | Slope/Plateau
cw = [0.22, 0.16, 0.14, 0.16, 0.16, 0.16]
assert abs(sum(cw) - 1.0) < 0.01
headers = ['Method', 'Origin', 'Entry Point',
           r'$\uparrow k_{\rm inf}$ effect',
           r'$\uparrow \rho_\mathcal{I}$ effect',
           'Slope / Plateau']
RW = R - L
row_h = (TOP - BOT) / (n + 1)

def cx(c): return L + sum(cw[:c]) * RW + cw[c] * RW / 2
def lx(c): return L + sum(cw[:c]) * RW

# family background bands
prev_fam = None
band_start = None
for i, (_, fam, *_) in enumerate(attacks):
    if fam != prev_fam:
        if prev_fam is not None:
            y0 = BOT + (n - i) * row_h
            y1 = BOT + (n - band_start) * row_h
            rect = FancyBboxPatch((L, y0), RW, y1 - y0,
                                  boxstyle='round,pad=0.003',
                                  fc=FAMILY_COLOR[prev_fam] + '1a',
                                  ec=FAMILY_COLOR[prev_fam] + '66', lw=0.8)
            ax.add_patch(rect)
        band_start = i
        prev_fam = fam
# last band
y0 = BOT
y1 = BOT + (n - band_start) * row_h
rect = FancyBboxPatch((L, y0), RW, y1 - y0,
                      boxstyle='round,pad=0.003',
                      fc=FAMILY_COLOR[prev_fam] + '1a',
                      ec=FAMILY_COLOR[prev_fam] + '66', lw=0.8)
ax.add_patch(rect)

# family left-edge bar
prev_fam = None
for i, (_, fam, *_) in enumerate(attacks):
    if fam != prev_fam:
        if prev_fam is not None:
            ax.plot([L, L], [BOT + (n-i)*row_h, BOT + (n-band_start)*row_h],
                    color=FAMILY_COLOR[prev_fam], lw=4, solid_capstyle='butt')
        band_start = i; prev_fam = fam
ax.plot([L, L], [BOT, BOT + (n-band_start)*row_h],
        color=FAMILY_COLOR[prev_fam], lw=4, solid_capstyle='butt')

# header row
ax.axhline(TOP, color='#333', lw=1.0)
for c, h in enumerate(headers):
    ax.text(cx(c), TOP + row_h*0.45, h, ha='center', va='center',
            fontsize=8, fontweight='bold', color='#222')

def badge(ax, x, y, w, h, text, fc, ec, fs=7.2, tc='white', bold=False):
    bw = w * 0.82; bh = h * 0.62
    rect = FancyBboxPatch((x - bw/2, y - bh/2), bw, bh,
                          boxstyle='round,pad=0.004', fc=fc, ec=ec, lw=0.6)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal')

for i, (name, fam, origin, entry, kinf, rho, sp) in enumerate(attacks):
    row = n - 1 - i
    y = BOT + row * row_h + row_h * 0.5
    w = cw[0] * RW; h = row_h

    # col 0: method name
    ax.text(cx(0), y, name, ha='center', va='center',
            fontsize=8, fontweight='bold', color=FAMILY_COLOR[fam])

    # col 1: origin
    oc = ORIGIN_COLOR.get(origin, '#888')
    if origin:
        badge(ax, cx(1), y, cw[1]*RW, h, ORIGIN_LABEL[origin],
              fc=oc+'33', ec=oc+'99', tc=oc, fs=7)

    # col 2: entry point
    ec2 = ENTRY_COLOR.get(entry)
    if entry and ec2:
        badge(ax, cx(2), y, cw[2]*RW, h, entry,
              fc=ec2+'33', ec=ec2+'99', tc=ec2, fs=7)
    elif entry:
        ax.text(cx(2), y, entry, ha='center', va='center', fontsize=7, color='#888')

    # col 3: k_inf
    if kinf:
        kc = C_DISSEM if 'User' in kinf else (C_EXTRACT if 'Item' in kinf else '#55A868')
        ax.text(cx(3), y, kinf, ha='center', va='center', fontsize=7.5, color=kc)

    # col 4: rho
    if rho and rho != 'Stable':
        rc = C_DISSEM if 'User' in rho else (C_EXTRACT if 'Item' in rho else '#55A868')
        ax.text(cx(4), y, rho, ha='center', va='center', fontsize=7.5, color=rc)
    elif rho == 'Stable':
        ax.text(cx(4), y, 'Stable', ha='center', va='center', fontsize=7, color='#aaa')

    # col 5: slope/plateau
    if sp == 'R':
        badge(ax, cx(5), y, cw[5]*RW, h, 'Reversed',
              fc=C_INVERT+'33', ec=C_INVERT+'99', tc=C_INVERT, fs=7.5, bold=True)
    elif sp == 'C':
        badge(ax, cx(5), y, cw[5]*RW, h, 'Consistent',
              fc=C_MONO+'22', ec=C_MONO+'77', tc=C_MONO, fs=7.5)

    # defense row: span cols 2-5
    if fam == 'Defense':
        note = {'G-Safeguard': 'denser graphs → stronger detection',
                'BlindGuard':  'robust across density variants',
                'T-Guard':     'denser graphs → stronger quarantine',
                'M-Guard':     'topology-agnostic (offline hardening)'}
        ax.text((lx(2) + R) / 2, y, note.get(name, ''), ha='center', va='center',
                fontsize=7, color='#666', style='italic')

    ax.axhline(BOT + row * row_h, color='#e0e0e0', lw=0.4)

# col separators
for c in range(1, len(cw)):
    ax.axvline(lx(c), color='#ddd', lw=0.4, ymin=BOT, ymax=TOP)

# legend
leg = [mpatches.Patch(fc=FAMILY_COLOR[f]+'55', ec=FAMILY_COLOR[f], label=f)
       for f in ['Dissem.', 'Extract.', 'Bidir.', 'Defense']]
leg += [mpatches.Patch(fc=C_INVERT+'44', ec=C_INVERT, label='Slope/plateau reversed'),
        mpatches.Patch(fc=C_MONO+'33',   ec=C_MONO,   label='Slope/plateau consistent')]
ax.legend(handles=leg, loc='lower center', bbox_to_anchor=(0.5, -0.10),
          ncol=6, fontsize=7, frameon=False, handlelength=1.2)

fig.savefig('figure/pdf/taxonomy_grid.pdf', bbox_inches='tight', dpi=200)
fig.savefig('figure/taxonomy_grid.png', bbox_inches='tight', dpi=200)
print("Saved grid")
plt.close(fig)

# ── FIGURE 2: compact single-column donut ────────────────────────────────────
# Single ring, labels outside with radial leader lines, grouped by family.
groups = [
    ('Dissem.',  ['NetSafe','CheatAgent','DrunkAgent','PI/CORBA','RecText']),
    ('Extract.', ['MAMA','InjecAgent','MASLeak']),
    ('Bidir.',   ['TOMA','MASTER']),
    ('Defense',  ['G-Safe','BlindGuard','T-Guard','M-Guard']),
]
total_segs = sum(len(g[1]) for g in groups)
GROUP_GAP_DEG = 8.0  # degrees gap between groups

fig2, ax2 = plt.subplots(figsize=(3.2, 3.2))
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_xlim(-1.7, 1.7); ax2.set_ylim(-1.7, 1.7)

full = 360.0
usable = full - len(groups) * GROUP_GAP_DEG
seg_deg = usable / total_segs

start = 90.0  # start at top, go clockwise (subtract angles)
for gname, names in groups:
    fc = FAMILY_COLOR[gname]
    n = len(names)
    group_span = n * seg_deg

    for j, name in enumerate(names):
        a0 = start - j * seg_deg
        a1 = a0 - seg_deg * 0.93
        amid = (a0 + a1) / 2

        # draw wedge
        theta_arr = np.linspace(np.radians(a1), np.radians(a0), 30)
        r_out, r_in = 1.0, 0.52
        xs = np.concatenate([r_in*np.cos(theta_arr), r_out*np.cos(theta_arr[::-1])])
        ys = np.concatenate([r_in*np.sin(theta_arr), r_out*np.sin(theta_arr[::-1])])
        ax2.fill(xs, ys, color=fc, alpha=0.85, linewidth=0.4, edgecolor='white')

        # label outside with short leader
        r_label = 1.28
        lx2 = r_label * np.cos(np.radians(amid))
        ly2 = r_label * np.sin(np.radians(amid))
        r_tip = 1.03
        tx = r_tip * np.cos(np.radians(amid))
        ty = r_tip * np.sin(np.radians(amid))
        ax2.annotate('', xy=(tx, ty), xytext=(lx2*0.93, ly2*0.93),
                     arrowprops=dict(arrowstyle='-', color=fc+'99', lw=0.6))
        ha = 'left' if lx2 > 0.05 else ('right' if lx2 < -0.05 else 'center')
        ax2.text(lx2, ly2, name, ha=ha, va='center',
                 fontsize=5.4, color='#222', fontweight='bold')

    # group arc label inside ring
    gmid_deg = start - group_span / 2 + seg_deg * 0.5
    gx = 0.76 * np.cos(np.radians(gmid_deg))
    gy = 0.76 * np.sin(np.radians(gmid_deg))
    ax2.text(gx, gy, gname, ha='center', va='center',
             fontsize=6, color='white', fontweight='bold',
             rotation=gmid_deg - 90 if gmid_deg > 0 else gmid_deg + 90)

    start -= group_span + GROUP_GAP_DEG

# center hole label
circle = plt.Circle((0, 0), 0.50, color='white', zorder=5)
ax2.add_patch(circle)
ax2.text(0, 0, 'MACF\nAttacks', ha='center', va='center',
         fontsize=7, fontweight='bold', color='#333', zorder=6)

leg2 = [mpatches.Patch(fc=FAMILY_COLOR[g[0]]+'cc', ec=FAMILY_COLOR[g[0]], label=g[0])
        for g in groups]
ax2.legend(handles=leg2, loc='lower center', bbox_to_anchor=(0.5, -0.08),
           ncol=2, fontsize=6, frameon=False, handlelength=1.0)

fig2.savefig('figure/pdf/taxonomy_wheel.pdf', bbox_inches='tight', dpi=200)
fig2.savefig('figure/taxonomy_wheel.png', bbox_inches='tight', dpi=200)
print("Saved wheel")
plt.close(fig2)
