import json
from typing import List, Tuple
import os
from datetime import datetime
import cv2


class COCOWriter:
    def __init__(
        self, input_dir: str, unity_annotations: List, output_dir: str, splits: List = [0.8, 0.1, 0.1]
    ) -> None:
        self.input_dir = input_dir
        self.unity_annotations = unity_annotations
        self.output_dir = output_dir
        self.splits = splits
        self.img_height, self.img_width, _ = self.get_image_dims()

    def get_image_dims(self) -> Tuple:
        """gets the size of the image. The assumption is that all the images are of the same size"""
        img_path = self.unity_annotations[0]["filename"]
        img = cv2.imread(os.path.join(self.input_dir, img_path))
        return img.shape

    def get_info_field(self):
        header = {}
        header["year"] = datetime.now().year
        header["version"] = "1.0"
        header["description"] = "Unity perception dataset"
        header["Contributor"] = "contributor"
        header["url"] = ""
        header["date_created"] = str(datetime.date(datetime.now()))
        return header

    def get_image_entry(self, annotation):
        entry = {}
        entry["id"] = annotation["id"]
        entry["width"] = self.img_width
        entry["height"] = self.img_height
        entry["filename"] = annotation["filename"]

        return entry

    def get_annotation_entry(self, annotation):
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

    def write(self):
        """
        input_dir: path to unity dataset
        unity_annotations: list of all the unity_annotations
        output_dir: directory in which to write
        splits: How to split the dataset. By default 80% is taken for training, 10 % for validation and 10% for testing
        """

        os.makedirs(os.path.join(self.output_dir, "annotations"), exist_ok=True)

        input_data_len = len(self.unity_annotations)
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
                data["images"].append(self.get_image_entry(annotation))
                data["annotations"].extend(self.get_annotation_entry(annotation))

            with open(os.path.join(self.output_dir, "annotations", f"{split}.json"), "w") as f:
                json.dump(data, f)

