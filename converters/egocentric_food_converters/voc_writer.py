from tqdm import tqdm
from typing import Dict, List
import os
from PIL import Image
from converters.egocentric_food_converters.egocentric_food_annotation import get_categories, load_annotations
from math import ceil


class VOCWriter:
    def __init__(self, input_dir: str, output_dir: str) -> None:
        """
        input_dir: path to unity dataset
        output_dir: directory in which to write
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.categories: Dict = get_categories(input_dir)

    def collect_annotations(self, split: str) -> Dict:
        """Since a bounding box is in a single line, this function collects all the annotations in a single dict.
            The assumption is that there is only one category per image"""

        collected_annotations = {}
        annotations = load_annotations(self.input_dir, split)
        print(f"Found {split} set of length {len(annotations)}")
        print("Loading annotations")
        for annotation in tqdm(annotations):
            orig_image_name = annotation[0].split("/")[-1]
            category_id = annotation[1]
            img = Image.open(os.path.join(self.input_dir, category_id, orig_image_name))
            img = img.convert("RGB")
            img_width, img_height = img.size
            filename = f"{category_id}_{orig_image_name}"  # use this filename to save
            if filename not in collected_annotations.keys():
                collected_annotations[filename] = {
                    "img_width": img_width,
                    "img_height": img_height,
                    "original_filename": orig_image_name,
                    "category": category_id,
                    "bounding_boxes": [],
                }

            collected_annotations[filename]["bounding_boxes"].append([ceil(float(x)) for x in annotation[2:]])

        return collected_annotations

    def write(self):
        """write the output dataset
        """
        os.makedirs(os.path.join(self.output_dir, "Annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "JPEGImages"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "ImageSets", "Main"), exist_ok=True)

        for split in ["val", "test", "train"]:
            annotations = self.collect_annotations(split)

            print("Writing annotations")
            for key, annotation in tqdm(annotations.items()):
                img = Image.open(
                    os.path.join(self.input_dir, annotation["category"], annotation["original_filename"])
                )
                img = img.convert("RGB")
                # add category id to filename so that images with the same name are not overwritten
                img.save(os.path.join(self.output_dir, "JPEGImages", key), "JPEG")
                data = f"""<annotation>
                        <floder>{os.path.split(self.output_dir)[-1]}</folder>
                        <filename>{key}</filename>
                        <source>
                            <database>Generated Data</database>
                            <annotation>PASCAL VOC2007</annotation>
                            <image>Egocentric Food</image>
                        </source>
                        <size>
                            <width>{annotation['img_width']}</width>
                            <height>{annotation['img_height']}</height>
                            <depth>3</depth>
                        </size>
                        """
                for obj in annotation["bounding_boxes"]:
                    data += f"""<object>
                                    <name>{self.categories[annotation['category']]}</name>
                                    <bndbox>
                                        <xmin>{int(obj[0])}</xmin>
                                        <ymin>{int(obj[1])}</ymin>
                                        <xmax>{int(obj[2])}</xmax>
                                        <ymax>{int(obj[3])}</ymax>
                                    </bndbox>
                                </object>
                            """

                data += "</annotation>"
                # write to annotation file
                with open(os.path.join(self.output_dir, "Annotations", key + ".xml"), "w") as annf:
                    annf.write(data)

                # write to category file
                with open(
                    os.path.join(self.output_dir, "ImageSets", "Main", f"{self.categories[annotation['category']]}_{split}.txt"), "a"
                ) as imgsetf:
                    imgsetf.write(f"{key} 1\n")
