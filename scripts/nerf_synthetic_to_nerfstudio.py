"""Convert a NeRF-synthetic scene (transforms_train.json + transforms_test.json)
to a Nerfstudio-format transforms.json that msplat can load.

Usage:
    python scripts/nerf_synthetic_to_nerfstudio.py lego lego_ns

Reads lego/transforms_train.json + lego/transforms_test.json + lego/{train,test}/*.png,
writes lego_ns/transforms.json + copies (or symlinks) images into lego_ns/images/.
"""

from __future__ import annotations
import json
import math
import os
import shutil
import sys
from PIL import Image


def convert(src: str, dst: str, symlink: bool = True):
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)

    frames_out = []
    W = H = None
    camera_angle_x = None

    for split in ("train", "test"):
        js_path = os.path.join(src, f"transforms_{split}.json")
        if not os.path.exists(js_path):
            print(f"skip: {js_path} not found")
            continue
        with open(js_path) as f:
            data = json.load(f)
        if camera_angle_x is None:
            camera_angle_x = data["camera_angle_x"]
        for frame in data["frames"]:
            rel = frame["file_path"]                        # "./train/r_0"
            base = os.path.basename(rel)                    # "r_0"
            src_img = os.path.join(src, split, f"{base}.png")
            if not os.path.exists(src_img):
                print(f"warn: image missing {src_img}, skipping")
                continue
            if W is None:
                im = Image.open(src_img)
                W, H = im.size
            new_name = f"{split}_{base}.png"                # collision-free
            dst_img = os.path.join(dst, "images", new_name)
            if not os.path.exists(dst_img):
                if symlink:
                    os.symlink(os.path.abspath(src_img), dst_img)
                else:
                    shutil.copy2(src_img, dst_img)
            frames_out.append({
                "file_path": f"images/{new_name}",
                "transform_matrix": frame["transform_matrix"],
            })

    assert W is not None, "No images found"
    fl = 0.5 * W / math.tan(0.5 * camera_angle_x)

    out = {
        "camera_model": "OPENCV",
        "fl_x": fl,
        "fl_y": fl,
        "cx": W / 2.0,
        "cy": H / 2.0,
        "w": W,
        "h": H,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": frames_out,
    }
    with open(os.path.join(dst, "transforms.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(frames_out)} frames to {dst}/transforms.json  ({W}x{H}, fl={fl:.2f})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
