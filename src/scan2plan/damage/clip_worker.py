"""Child-process CLIP zero-shot crop classifier: python -m scan2plan.damage.clip_worker <list.txt> <out.npy>

Output (N, n_damage_classes + 1): per-class probability (prompt-ensembled), last column = best
negative. Model: openai/clip-vit-base-patch32 (MIT), fetched into weights/hf.
"""
import os
import sys

import numpy as np


def main(lst, out):
    os.environ.setdefault("HF_HOME", os.path.abspath("weights/hf"))
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from transformers.utils import logging

    from scan2plan.damage.detect import CLASSES, NEGATIVES

    logging.set_verbosity_error()
    mid = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(mid).eval()
    proc = CLIPProcessor.from_pretrained(mid)
    texts, owner = [], []
    for k, (c, ps) in enumerate(CLASSES.items()):
        for p in ps:
            texts.append("a photo of " + p)
            owner.append(k)
    for p in NEGATIVES:
        texts.append("a photo of " + p)
        owner.append(len(CLASSES))
    owner = np.array(owner)
    paths = [p.strip() for p in open(lst) if p.strip()]
    res = []
    with torch.no_grad():
        tin = proc(text=texts, return_tensors="pt", padding=True)
        tf = model.get_text_features(**tin)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        for i in range(0, len(paths), 32):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + 32]]
            iin = proc(images=ims, return_tensors="pt")
            f = model.get_image_features(**iin)
            f = f / f.norm(dim=-1, keepdim=True)
            pr = (100 * f @ tf.T).softmax(-1).numpy()
            # aggregate prompts: class prob = sum over its prompts
            agg = np.stack([pr[:, owner == k].sum(1) for k in range(len(CLASSES) + 1)], 1)
            res.append(agg)
    np.save(out, np.concatenate(res) if res else np.zeros((0, len(CLASSES) + 1)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
