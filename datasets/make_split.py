from pathlib import Path
import random, shutil

ROOT = Path("data/celeba_hq_128")
SIZES = {"train": 26_000, "val": 2_000, "test": 2_000}
SEED = 447

random.seed(SEED)
all_imgs = sorted(ROOT.glob("*.png"))
random.shuffle(all_imgs)

offset = 0
for split, n in SIZES.items():
    out_dir = ROOT.parent / split
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = all_imgs[offset : offset + n]
    offset += n
    for p in subset:
        shutil.move(str(p), out_dir / p.name)
    print(f"{split}: {len(subset)} images -> {out_dir}")

assert offset == len(all_imgs), "sizes don't sum to total image count"
print("Done.")
