import json
import math
import random
import sys

EOS = "eos1234"
LAM = 1.0   # weight for the grammar (4th) dimension

# Load 4D locations
with open("corpus.json", "r") as f:
    data = json.load(f)

locs = {w["word"]: tuple(w["loc"]) for w in data["words"]}
vocab = list(locs.keys())

def dist4(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    dg = a[3] - b[3]
    return math.sqrt(dx*dx + dy*dy + dz*dz + LAM*dg*dg)

def next_word(context, eps=1e-3):
    weights = []
    for w in vocab:
        dists = [dist4(locs[w], locs[c]) for c in context]
        avg = (sum(dists) / len(dists)) if dists else eps
        weights.append(1.0 / (avg + eps))
    total = sum(weights)
    probs = [wgt / total for wgt in weights]
    return random.choices(vocab, probs, k=1)[0]

def generate(seed, max_len=500, window=20):
    gen = list(seed)
    for _ in range(max_len):
        ctx = gen[-window:]
        nxt = next_word(ctx)
        if nxt == EOS:
            break
        gen.append(nxt)
    return gen

def main():
    print("Enter seed words (space-separated):")
    seed = sys.stdin.readline().strip().split()
    # filter out-of-vocab and EOS
    seed = [w for w in seed if w in vocab and w != EOS]
    if not seed:
        print("No valid seed words in vocab. Exiting.")
        return

    seq = generate(seed)
    print(" ".join(seq))

if __name__ == "__main__":
    main()
