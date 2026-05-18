"""Reproducibly fetch the public CC0 idiom dictionary.

The original build_idiom_source.py depended on a local chinese-dictionary
dump that was deleted. The equivalent public CC0 source is
pwxcoo/chinese-xinhua `data/idiom.json` (30,895 idioms, ~24k with a
`derivation`/出处). Pinned to a specific commit + sha256 so regeneration is
deterministic and auditable.

Usage:  python scripts/fetch_idiom_source_data.py
Output: data/_vendor/xinhua_idiom.json  (git-ignored; large, re-fetchable)
"""
from __future__ import annotations
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PIN_COMMIT = "8de1001aa499cd65fb97ef8712550188c0297a08"
URL = (f"https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/"
       f"{PIN_COMMIT}/data/idiom.json")
SHA256 = "1d4b4f454ce1c416d6a1ab2369d6e66c0ff99e04390172eef70790499e21ce19"
OUT = REPO / "data" / "_vendor" / "xinhua_idiom.json"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and hashlib.sha256(OUT.read_bytes()).hexdigest() == SHA256:
        print(f"already present & verified → {OUT.relative_to(REPO)}")
        return 0
    print(f"fetching {URL}")
    data = urllib.request.urlopen(URL, timeout=60).read()  # noqa: S310
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        print(f"ERROR sha256 mismatch\n  expected {SHA256}\n  got      {got}")
        return 1
    OUT.write_bytes(data)
    print(f"verified & wrote {len(data):,} bytes → {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
