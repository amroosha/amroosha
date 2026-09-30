# -*- coding: utf-8 -*-
"""Generate BrailleArt.svg.

Pipeline:
  1. read the original braille block from git history (commit 5b5eb8d)
  2. normalise line widths (the source row that used ASCII spaces is realigned)
  3. morphological SMOOTHING at dot level: closing by reconstruction (3x3)
     then opening by reconstruction (3x3) - fills pinholes/notches and
     deletes all floating speck noise while preserving every thin stroke
  4. render as a geometric dot grid inside the animated smoke frame
"""
import pathlib
import re
import subprocess
from collections import Counter

import numpy as np
from scipy import ndimage as ndi

BASE = pathlib.Path(__file__).resolve().parent
OUT = BASE / "BrailleArt.svg"

CW, CH, R = 12, 24, 2.9       # cell width, cell height, dot radius
PAD = 48                       # panel padding around the art
RADIUS = 30                    # panel corner radius
SE3 = np.ones((3, 3), bool)
S8 = np.ones((3, 3), bool)

SR_BITS = [(0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80)]
BITS = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (1, 0): 0x08,
        (1, 1): 0x10, (1, 2): 0x20, (0, 3): 0x40, (1, 3): 0x80}


def load_rows():
    src = subprocess.run(['git', 'show', '5b5eb8d:README.md'],
                         capture_output=True, cwd=BASE).stdout.decode('utf-8')
    m = re.search(r'Braille Art.*?```\r?\n(.*?)```', src, re.S)
    if not m:
        raise SystemExit('braille code block not found in git history')
    lines = [l.rstrip('\r') for l in m.group(1).split('\n') if l.strip()]
    rows = []
    for line in lines:
        bits = []
        for ch in line:
            o = ord(ch)
            bits.append(o - 0x2800 if 0x2800 <= o <= 0x28FF else 0)
        rows.append(bits)

    # row 4 of the source carries ASCII spaces instead of braille blanks and
    # is 3 cells too wide -> realign to the most common line length
    target = Counter(len(r) for r in rows).most_common(1)[0][0]
    for r in rows:
        while len(r) > target:
            best = best_len = cur = cur_start = -1
            for i, v in enumerate(r + [1]):
                if v == 0:
                    if cur < 0:
                        cur, cur_start = i, i
                else:
                    if cur >= 0 and i - cur > best_len:
                        best, best_len, cur_start = cur_start, i - cur, cur_start
                    cur = -1
            del r[best + best_len // 2]
        while len(r) < target:
            r.append(0)
    return rows, target, len(rows)


def open_recon(x, se):
    seed = ndi.binary_erosion(x, structure=se)
    return ndi.binary_propagation(seed, mask=x, structure=S8)


def close_recon(x, se):
    return ~open_recon(~x, se)


def smooth(rows, width, height):
    grid = np.zeros((height * 4, width * 2), dtype=bool)
    for y, row in enumerate(rows):
        for x, mask in enumerate(row):
            if not mask:
                continue
            for (col, sub), bit in BITS.items():
                if mask & bit:
                    grid[y * 4 + sub, x * 2 + col] = True
    before = int(grid.sum())
    grid = open_recon(close_recon(grid, SE3), SE3)   # close then open
    after = int(grid.sum())
    lab, n = ndi.label(grid, structure=S8)
    out = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            mask = 0
            for (col, sub), bit in BITS.items():
                if grid[y * 4 + sub, x * 2 + col]:
                    mask |= bit
            out[y][x] = mask
    print('smoothing: ink %d -> %d dots, components %d -> %d'
          % (before, after, ndi.label(grid, structure=S8)[1], n))
    return out


def art_shapes(rows, width, x0, y0):
    shapes = []
    for y, row in enumerate(rows):
        base_y = y0 + y * CH
        for sr, (b0, b1) in enumerate(SR_BITS):
            cy = base_y + 3 + sr * 6

            def has(p):
                cell, col = divmod(p, 2)
                return bool(row[cell] & (b0 if col == 0 else b1))

            p = 0
            while p < 2 * width:
                if has(p):
                    start = p
                    while p < 2 * width and has(p):
                        p += 1
                    end = p - 1
                    sx = x0 + (start // 2) * CW + (3 if start % 2 == 0 else 9)
                    ex = x0 + (end // 2) * CW + (3 if end % 2 == 0 else 9)
                    x = sx - R
                    w = (ex - sx) + 2 * R
                    shapes.append(
                        f'<rect x="{x:g}" y="{cy - R:g}" width="{w:g}" '
                        f'height="{2 * R:g}" rx="{R:g}"/>'
                    )
                else:
                    p += 1
    return shapes


def main():
    rows, width, height = load_rows()
    rows = smooth(rows, width, height)
    pw = width * CW + PAD * 2
    ph = height * CH + PAD * 2
    x0 = y0 = PAD
    art_h = height * CH
    shapes = art_shapes(rows, width, x0, y0)

    def blob(cx, cy, rx, ry, fill, dur, dx, delay):
        return (
            f'<ellipse cx="{cx:g}" cy="{cy:g}" rx="{rx:g}" ry="{ry:g}" fill="{fill}">'
            f'<animate attributeName="cx" values="{cx:g};{cx + dx:g};{cx:g}" '
            f'dur="{dur}s" begin="{delay}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{cy:g};{cy - dx * 0.4:g};{cy:g}" '
            f'dur="{dur * 1.3:g}s" begin="{delay}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0.55;0.95;0.55" '
            f'dur="{dur * 0.8:g}s" begin="{delay}s" repeatCount="indefinite"/>'
            f'</ellipse>'
        )

    smoke = "".join([
        blob(pw * 0.30, ph * 0.34, pw * 0.34, ph * 0.16, "#6E87A6", 26, 90, 0),
        blob(pw * 0.68, ph * 0.58, pw * 0.36, ph * 0.14, "#5D7392", 33, -80, -8),
        blob(pw * 0.45, ph * 0.80, pw * 0.40, ph * 0.13, "#7C93B0", 40, 70, -15),
    ])

    frame_rect = f'x="4" y="4" width="{pw - 8}" height="{ph - 8}" rx="{RADIUS - 4}"'

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{pw}" height="{ph}" viewBox="0 0 {pw} {ph}" role="img" aria-label="Braille art in an animated frame">
  <title>Braille art</title>
  <defs>
    <linearGradient id="panel" x1="0" y1="0" x2="0.35" y2="1">
      <stop offset="0" stop-color="#151D29"/>
      <stop offset="0.55" stop-color="#0F151E"/>
      <stop offset="1" stop-color="#0A0E14"/>
    </linearGradient>
    <linearGradient id="ink" gradientUnits="userSpaceOnUse" x1="0" y1="{y0}" x2="0" y2="{y0 + art_h}">
      <stop offset="0" stop-color="#FFFFFF"/>
      <stop offset="0.55" stop-color="#DEE7F1"/>
      <stop offset="1" stop-color="#A7B6C8"/>
    </linearGradient>
    <linearGradient id="edge" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#38BDF8"/>
      <stop offset="0.5" stop-color="#7C93FC"/>
      <stop offset="1" stop-color="#A78BFA"/>
    </linearGradient>
    <radialGradient id="haze" cx="0.5" cy="0.28" r="0.75">
      <stop offset="0" stop-color="#Bcd3ea" stop-opacity="0.10"/>
      <stop offset="1" stop-color="#Bcd3ea" stop-opacity="0"/>
    </radialGradient>
    <filter id="smoke" x="-70%" y="-70%" width="240%" height="240%">
      <feTurbulence type="fractalNoise" baseFrequency="0.009 0.017" numOctaves="3" seed="11" result="n"/>
      <feDisplacementMap in="SourceGraphic" in2="n" scale="110" xChannelSelector="R" yChannelSelector="G"/>
      <feGaussianBlur stdDeviation="16"/>
    </filter>
    <filter id="halo" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="0" stdDeviation="7" flood-color="#38BDF8" flood-opacity="0.35"/>
    </filter>
    <filter id="soft" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="5"/>
    </filter>
    <clipPath id="panelClip">
      <rect x="0" y="0" width="{pw}" height="{ph}" rx="{RADIUS}"/>
    </clipPath>
  </defs>

  <g clip-path="url(#panelClip)">
    <rect width="{pw}" height="{ph}" fill="url(#panel)"/>
    <rect width="{pw}" height="{ph}" fill="url(#haze)"/>
    <g filter="url(#smoke)" opacity="0.5">{smoke}</g>
    <g fill="url(#ink)" filter="url(#halo)">
{chr(10).join("      " + s for s in shapes)}
    </g>
    <g fill="#8FB6E8" opacity="0.10" filter="url(#smoke)">
      <ellipse cx="{pw * 0.5:g}" cy="{y0 + art_h * 0.35:g}" rx="{pw * 0.30:g}" ry="{ph * 0.10:g}"/>
    </g>
  </g>

  <rect {frame_rect} fill="none" stroke="url(#edge)" stroke-width="8"
        opacity="0.30" filter="url(#soft)" stroke-dasharray="30 16">
    <animate attributeName="stroke-dashoffset" from="0" to="-92" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.22;0.5;0.22" dur="5s" repeatCount="indefinite"/>
  </rect>
  <rect {frame_rect} fill="none" stroke="url(#edge)" stroke-width="3"
        stroke-dasharray="30 16" stroke-linecap="round">
    <animate attributeName="stroke-dashoffset" from="0" to="-92" dur="4s" repeatCount="indefinite"/>
  </rect>
</svg>
'''
    OUT.write_text(svg, encoding='utf-8')
    print('rows=%d cols=%d panel=%dx%d shapes=%d bytes=%d'
          % (height, width, pw, ph, len(shapes), len(svg.encode('utf-8'))))


if __name__ == '__main__':
    main()
