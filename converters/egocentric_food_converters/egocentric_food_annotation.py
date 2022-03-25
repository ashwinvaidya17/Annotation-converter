"""Functions for loading egocentric food annotation"""
from typing import Dict, List
import os


def load_annotations(data_path: str, split: str) -> List:
    annotations = []

    with open(os.path.join(data_path, f"{split}_list.txt"), "r") as f:
        data = f.readlines()
        # split each column in a line and store in a separate list
        annotations = [a.split() for a in data]

    # first line contains column names which we are not interested in
    return annotations[1:]


def get_categories(data_path: str) -> Dict:
    categories = {}
    with open(os.path.join(data_path, "category.txt"), "r") as f:
        data = f.readlines()
        for entry in data:
            k, v = entry.split()
            categories[k] = v

        categories.pop("id")  # remove the headers
    return categories
