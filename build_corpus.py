#!/usr/bin/env python3
import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(
        description="Build corpus.json with 4D (x,y,z,g) locations"
    )
    p.add_argument("xml_file", help="Input XML file (data.xml)")
    p.add_argument(
        "--max-coord",
        type=float,
        default=100.0,
        help="Max value for x,y,z coords (uniform in [0,max])"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    p.add_argument(
        "-o",
        "--out",
        default="corpus.json",
        help="Output JSON filename"
    )
    return p.parse_args()

def tokenize(text):
    # simple word tokenizer: letters, digits, underscore
    return re.findall(r"\b\w+\b", text.lower())

def main():
    args = parse_args()
    random.seed(args.seed)

    # Parse XML and collect input/output token lists
    tree = ET.parse(args.xml_file)
    root = tree.getroot()

    pairs = []      # list of (inp_tokens, out_tokens)
    vocab = set()
    for pair in root.findall(".//pair"):
        inp = pair.findtext("input", "")
        out = pair.findtext("output", "")
        itoks = tokenize(inp)
        otoks = tokenize(out)
        pairs.append((itoks, otoks))
        vocab.update(itoks)
        vocab.update(otoks)

    # Build bigram counts: bigram[a][b] = count of (a->b)
    bigram = defaultdict(lambda: defaultdict(int))
    for itoks, otoks in pairs:
        for seq in (itoks, otoks):
            for a, b in zip(seq, seq[1:]):
                bigram[a][b] += 1

    # Compute conditional P(b|a)
    cond = {}
    for a, followers in bigram.items():
        total = sum(followers.values())
        cond[a] = {b: c / total for b, c in followers.items()}

    # Compute g(w) = average P(w|prev) over all prev
    g = {}
    for w in vocab:
        preds = [a for a in cond if w in cond[a]]
        if not preds:
            g[w] = 0.0
        else:
            g[w] = sum(cond[a][w] for a in preds) / len(preds)

    # Normalize g to [0,1] by dividing by max_g (if max_g>0)
    maxg = max(g.values()) if g else 1.0
    if maxg > 0:
        for w in g:
            g[w] /= maxg

    # Assign random x,y,z in [0,max_coord] and attach g(w)
    words = []
    for w in sorted(vocab):
        x = random.uniform(0, args.max_coord)
        y = random.uniform(0, args.max_coord)
        z = random.uniform(0, args.max_coord)
        words.append({"word": w, "loc": [x, y, z, g[w]]})

    # Write out corpus.json
    with open(args.out, "w") as f:
        json.dump({"words": words}, f, indent=2)

    print(f"Wrote {len(words)} words to {args.out}")

if __name__ == "__main__":
    main()
