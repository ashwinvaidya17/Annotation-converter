"""Common functions for handling Unity Annotations"""

import glob
import json
import os
from typing import Dict, List, Tuple

import cv2


def load_unity_annotations(data_path) -> List:
    annotations = []

    annotation_files = glob.glob(os.path.join(data_path, "Dataset*", "captures*.json"))
    for annotation_file in annotation_files:
        with open(annotation_file, "r") as f:
            annotation = json.load(f)
        annotations.extend(annotation["captures"])

    return annotations


def get_image_dims(input_dir: str, annotations: List[Dict]) -> Tuple:
    """gets the size of the image. The assumption is that all the images are of the same size
    
    :return: Tuple(width, height, channels)
    """
    img_path = annotations[0]["filename"]
    img = cv2.imread(os.path.join(input_dir, img_path))
    return img.shape


def get_image_entry(img_width: int, img_height: int, annotation: Dict):
    entry = {}
    entry["id"] = annotation["id"]
    entry["width"] = img_width
    entry["height"] = img_height
    entry["filename"] = annotation["filename"]

    return entry


def get_annotation_entry(annotation: Dict):
    entries = []
    for anns in annotation["annotations"]:
        entry = {}
        entry["image_id"] = annotation["id"]
        entry["bbox"] = [
            anns["values"][0]["x"],
            anns["values"][0]["y"],
            anns["values"][0]["width"],
            anns["values"][0]["height"],
        ]
        entry["id"] = anns["id"]
        entry["category_id"] = anns["values"][0]["label_id"]

        entries.append(entry)
    return entries
