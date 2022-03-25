import os
from typing import List

from PIL import Image
from tqdm import tqdm

from converters.unity_perception_converters import unity_annotations


class VOCWriter:
    def __init__(self, input_dir: str, output_dir: str, splits: List = [0.8, 0.1, 0.1]) -> None:
        """
        input_dir: path to unity dataset
        output_dir: directory in which to write
        splits: How to split the dataset. By default 80% is taken for training, 10 % for validation and 10% for testing
        """
        self.input_dir = input_dir
        self.unity_annotations = unity_annotations.load_unity_annotations(self.input_dir)
        self.output_dir = output_dir
        self.splits = splits
        self.img_height, self.img_width, _ = unity_annotations.get_image_dims(self.input_dir, self.unity_annotations)

    def write(self):
        """write the output dataset
        """

        print(f"Found dataset of size {len(self.unity_annotations)}")

        os.makedirs(os.path.join(self.output_dir, "Annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "JPEGImages"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "ImageSets", "Main"), exist_ok=True)

        input_data_len = len(self.unity_annotations)
        # TODO add shuffle to splits
        train_split = self.unity_annotations[: int(self.splits[0] * input_data_len)]
        val_split = self.unity_annotations[
            int(self.splits[0] * input_data_len) : int(self.splits[0] * input_data_len)
            + int(self.splits[1] * input_data_len)
        ]
        test_split = self.unity_annotations[
            int((self.splits[0] + self.splits[1]) * input_data_len) : int(
                (self.splits[0] + self.splits[1]) * input_data_len
            )
            + int(self.splits[2] * input_data_len)
        ]

        for split in ["train", "val", "test"]:
            if split == "train":
                annotations = train_split
            elif split == "val":
                annotations = val_split
            else:
                annotations = test_split

            for annotation in tqdm(annotations):
                img = Image.open(os.path.join(self.input_dir, annotation["filename"]))
                filename = os.path.split(annotation["filename"])[-1].split(".")[0]
                img = img.convert("RGB")
                # save image as jpeg. Not really necessary.
                img.save(os.path.join(self.output_dir, "JPEGImages", f"{filename}.jpg"), "JPEG")
                data = f"""<annotation>
                        <floder>{os.path.split(self.output_dir)[-1]}</folder>
                        <filename>{os.path.split(annotation['filename'])[-1]}
                        <source>
                            <database>Generated Data</database>
                            <annotation>PASCAL VOC2007</annotation>
                            <image>unity perception</image>
                        </source>
                        <size>
                            <width>{self.img_width}</width>
                            <height>{self.img_height}</height>
                            <depth>3</depth>
                        </size>
                        """
                categories = set()
                for obj in annotation["annotations"][0]["values"]:
                    categories.add(obj["label_name"])
                    data += f"""<object>
                                    <name>{obj['label_name']}</name>
                                    <bndbox>
                                        <xmin>{int(obj['x'])}</xmin>
                                        <ymin>{int(obj['y'])}</ymin>
                                        <xmax>{int(obj['x'] + obj['width'])}</xmax>
                                        <ymax>{int(obj['y'] + obj['height'])}</ymax>
                                    </bndbox>
                                </object>
                            """

                data += "</annotation>"
                # write to annotation file
                with open(os.path.join(self.output_dir, "Annotations", filename + ".xml"), "w") as annf:
                    annf.write(data)

                # write to category file
                for category in categories:
                    with open(
                        os.path.join(self.output_dir, "ImageSets", "Main", f"{category}_{split}.txt"), "a"
                    ) as imgsetf:
                        imgsetf.write(f"{filename} 1\n")
