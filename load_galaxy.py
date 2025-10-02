from pathlib import Path

from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

save_path = Path("./data/galaxy")
save_path.mkdir(parents=True, exist_ok=True)

# ds = load_dataset("matthieulel/galaxy10_decals")

# ds.save_to_disk(save_path)
ds = load_from_disk(save_path)

label_names = sorted(ds["train"].unique('label'))

def export_split(split_name: str, root: str = "galaxy10_decals_imgs"):
    split = ds[split_name]
    root = Path(root) / split_name
    root.mkdir(parents=True, exist_ok=True)

    # Make class subdirs
    class_dirs = [root / str(name) for name in label_names]
    for d in class_dirs:
        d.mkdir(parents=True, exist_ok=True)

    for i, example in tqdm(enumerate(split), total=len(split), desc=f"Exporting {split_name}"):
        img = example["image"]         # PIL.Image (from datasets 'Image' feature)
        label_idx = label_names.index(int(example["label"]))
        class_dir = class_dirs[label_idx]
        img_path = class_dir / f"{split_name}_{i:06d}.png"
        # Ensure RGB and save
        img.save(img_path)

# Uncomment to export (this will write ~17k PNGs, ~GBs of data)
export_split("train")
export_split("test")