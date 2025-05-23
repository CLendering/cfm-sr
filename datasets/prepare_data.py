from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm


def resize_to_128(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [f for f in input_dir.glob("*") if f.suffix.lower() in [".jpg", ".png"]]
    )

    for file in tqdm(files, desc="Resizing to 128x128"):
        img = Image.open(file).convert("RGB")
        img_resized = TF.resize(img, 128, interpolation=Image.BICUBIC)
        fname = file.stem.zfill(5) + ".png"
        img_resized.save(output_dir / fname, format="PNG")


if __name__ == "__main__":
    input_path = "celeba_hq_256"
    output_path = "celeba_hq_128"
    resize_to_128(input_path, output_path)
