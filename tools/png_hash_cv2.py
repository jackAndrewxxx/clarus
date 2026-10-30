"""Reference checksums of cv2-decoded images (RGB order) to verify the Rust PNG decoder.

  clarus png-hash DIR > a.txt ; python tools/png_hash_cv2.py DIR > b.txt ; diff a.txt b.txt
"""
import os
import sys

import cv2
import numpy as np

for d in sys.argv[1:]:
    for name in sorted(os.listdir(d)):
        if not name.lower().endswith(".png"):
            continue
        img = np.ascontiguousarray(cv2.imread(os.path.join(d, name), cv2.IMREAD_COLOR)[:, :, ::-1]).reshape(-1)
        idx = np.arange(img.size, dtype=np.uint64)
        w = ((idx * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) + np.uint64(1)
        h = int((img.astype(np.uint64) * w).sum(dtype=np.uint64))
        print(name, f"{img.size // 3 and cv2.imread(os.path.join(d, name)).shape[1]}x"
                    f"{cv2.imread(os.path.join(d, name)).shape[0]}", f"{h:016x}")
