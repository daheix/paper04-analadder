#!/usr/bin/env python3
"""P04 S5.3 duplication self-check: 8-gram overlap vs sibling manuscripts.

Extracts detex'd text from each manuscript's main.tex, lowercases,
strips non-alphanumerics, and reports the longest shared 8-gram count.
Independence discipline (SKILL.md): manuscripts must not cite each
other; near-zero 8-gram overlap is the textual counterpart.
"""
import re
import subprocess
import sys
from pathlib import Path

SCI = Path(__file__).resolve().parents[2]


def text_of(tex):
    try:
        out = subprocess.run(["detex", str(tex)], capture_output=True, text=True, timeout=60)
        raw = out.stdout or Path(tex).read_text()
    except Exception:
        raw = Path(tex).read_text()
    s = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*", " ", raw)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return s.split()


def grams(tokens, n=8):
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def main():
    p04 = grams(text_of(SCI / "paper04_analadder" / "manuscript" / "main.tex"))
    for sib in ["paper02_torque", "paper03_protocol", "paper01_rebuild"]:
        tex = SCI / sib / "manuscript" / "main.tex"
        if not tex.exists():
            continue
        g = grams(text_of(tex))
        inter = p04 & g
        print(f"P04 vs {sib}: |8-gram| P04={len(p04)} sib={len(g)} overlap={len(inter)}")
        for x in sorted(inter)[:5]:
            print("   shared:", x)
    sys.exit(0)


if __name__ == "__main__":
    main()
