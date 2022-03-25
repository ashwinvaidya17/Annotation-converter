"""Generates a simple annotation format for detection

output:
    annotations:
        001.txt
        002.txt
        .

    images:
        001.jpg
        002.jpg
        .
    
    dataset.csv

annotation format
class_id x1 y1 x2 y2

dataset.csv
img_num.jpg, label_num.txt
"""


from converters.unity_perception_converters import unity_annotations
import os
from tqdm import tqdm
from PIL import Image
from glob import glob
import math


class SimplifiedDetectionWriter:
    def __init__(self, input_dir: str, output_dir: str):
        """

        Args:
            input_dir (str): path to input dir
            output_dir (str): path to output dir
        """

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.unity_annotations = unity_annotations.load_unity_annotations(self.input_dir)

    def write(self):
        """Write the output dataset
        """

        print(f"Found dataset of size {len(self.unity_annotations)}")
        precision = math.ceil(math.log(len(self.unity_annotations), 10))

        os.makedirs(os.path.join(self.output_dir, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

        annotation_counter = 0
        with open(os.path.join(self.output_dir, "dataset.csv"), "w") as csv_file:
            for annotation in tqdm(self.unity_annotations):
                img = Image.open(os.path.join(self.input_dir, annotation["filename"]))
                img.save(os.path.join(self.output_dir, "images", f"{annotation_counter:0{precision}}.png"), "PNG")
                with open(
                    os.path.join(self.output_dir, "annotations", f"{annotation_counter:0{precision}}.txt"), "w"
                ) as ann_file:
                    entries = unity_annotations.get_annotation_entry(annotation)
                    for entry in entries:
                        ann_file.write(f"{entry['category_id']} {' '.join([str(num) for num in entry['bbox']])}\n")

                csv_file.write(f"{annotation_counter:0{precision}}.png,{annotation_counter:0{precision}}.txt\n")
                annotation_counter += 1
