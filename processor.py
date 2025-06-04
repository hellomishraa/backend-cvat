import json
from pycocotools.coco import COCO
import numpy as np
import cv2
import requests
from utils import array_encode, get_image_dimensions_from_url, get_mask_center

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_categories(file_path):
    data = load_json(file_path)
    return [cat['name'] for cat in data['categories']]

def run_processing(coco_file, category_filter, filename_to_url):
    coco_data = load_json(coco_file)
    coco = COCO(coco_file)

    id_to_img = {img['id']: img for img in coco_data['images']}
    id_to_cat = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_ids = [cid for cid, cname in id_to_cat.items() if not category_filter or cname in category_filter]

    output = {}
    stats = {'total_images': len(coco_data['images']), 'processed_images': 0, 'processed_masks': 0, 'categories': {}}
    
    for img in coco_data['images']:
        img_id = img['id']
        file_name = img['file_name']
        image_url = filename_to_url.get(file_name, file_name)

        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=category_ids)
        if not ann_ids:
            continue

        output.setdefault(image_url, {"centres": {}, "masks": {}})
        category_counters = {}

        for ann_id in ann_ids:
            ann = coco.loadAnns(ann_id)[0]
            original_cat = id_to_cat[ann['category_id']]
            cat_name = "walls" if original_cat == "wall" else original_cat

            if category_filter and cat_name not in category_filter:
                continue

            category_counters.setdefault(cat_name, 0)
            category_counters[cat_name] += 1
            index = str(category_counters[cat_name])

            mask = coco.annToMask(ann)
            mask = (mask * 255).astype(np.uint8)
            b64 = array_encode(mask)
            center = get_mask_center(mask)

            output[image_url]["masks"].setdefault(cat_name, {})[index] = b64
            if center:
                output[image_url]["centres"].setdefault(cat_name, {})[index] = center

            stats['processed_masks'] += 1
            stats['categories'][cat_name] = stats['categories'].get(cat_name, 0) + 1

        stats['processed_images'] += 1

    return {"output": output, "stats": stats}