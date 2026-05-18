"""Reproducibly fetch the public CC0 dictionaries that replace the deleted
local chinese-dictionary dump.

Both come from pwxcoo/chinese-xinhua, each pinned to a specific commit +
sha256 so regeneration is deterministic and auditable:

  - idiom.json — 30,895 idioms, ~24k with a `derivation`/出处.
    Used by build_idiom_source_v2.py.
  - word.json  — 16,142 单字 entries; each `explanation` states the 本义
    in a parenthetical. Used by regen_char_gloss_candidates.py to give the
    18 circular-gold ("同本义。") items a real, verifiable modern gloss.

Usage:  python scripts/fetch_idiom_source_data.py
Output: data/_vendor/*.json  (git-ignored; large, re-fetchable)
"""
from __future__ import annotations
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "data" / "_vendor"

# name → (pinned commit, sha256)
ASSETS = {
    "idiom.json": ("8de1001aa499cd65fb97ef8712550188c0297a08",
                   "1d4b4f454ce1c416d6a1ab2369d6e66c0ff99e04390172eef70790499e21ce19"),
    "word.json":  ("cad702f2fbf4fa28755821772fbbe8525b6771b0",
                   "8ae3453eacc5b0f3fdfba47eac8bb686cd73914d278e8b851ae6ef81082f80e7"),
}


def fetch(name: str, commit: str, sha256: str) -> bool:
    out = VENDOR / f"xinhua_{name}"
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == sha256:
        print(f"already verified → {out.relative_to(REPO)}")
        return True
    url = (f"https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/"
           f"{commit}/data/{name}")
    print(f"fetching {url}")
    data = urllib.request.urlopen(url, timeout=60).read()  # noqa: S310
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        print(f"ERROR sha256 mismatch for {name}\n"
              f"  expected {sha256}\n  got      {got}")
        return False
    out.write_bytes(data)
    print(f"verified & wrote {len(data):,} bytes → {out.relative_to(REPO)}")
    return True


def main() -> int:
    VENDOR.mkdir(parents=True, exist_ok=True)
    ok = all(fetch(n, c, s) for n, (c, s) in ASSETS.items())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
