import json
import os
from datetime import datetime
from typing import List, Tuple

from converters.unity_perception_converters import unity_annotations


class COCOWriter:
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

    def get_info_field(self):
        header = {}
        header["year"] = datetime.now().year
        header["version"] = "1.0"
        header["description"] = "Unity perception dataset"
        header["Contributor"] = "contributor"
        header["url"] = ""
        header["date_created"] = str(datetime.date(datetime.now()))
        return header

    def write(self):

        os.makedirs(os.path.join(self.output_dir, "annotations"), exist_ok=True)

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

            data = {}
            data["info"] = self.get_info_field()
            data["images"] = []
            data["annotations"] = []

            for annotation in annotations:
                data["images"].append(unity_annotations.get_image_entry(self.img_width, self.img_height, annotation))
                data["annotations"].extend(unity_annotations.get_annotation_entry(annotation))

            with open(os.path.join(self.output_dir, "annotations", f"{split}.json"), "w") as f:
                json.dump(data, f)
