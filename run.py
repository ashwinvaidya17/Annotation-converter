import argparse
import glob
import json
import os

from converters.coco_writer import COCOWriter


def load_unity_annotations(data_path):
    annotations = []

    annotation_files = glob.glob(os.path.join(data_path, "Dataset*", "captures*.json"))
    for annotation_file in annotation_files:
        with open(annotation_file, "r") as f:
            annotation = json.load(f)
        annotations.extend(annotation["captures"])

    return annotations


def converter(args):
    unity_annotations = load_unity_annotations(args.input)

    output_format = args.output_format
    if output_format == "coco":
        writer = COCOWriter(args.input, unity_annotations, args.output_dir)
    else:
        raise NotImplementedError(f"Output format {output_format} not supported yet")

    writer.write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert data from Unity annotation to selected annotation format.")
    parser.add_argument("--input", type=str, help="Path to unity dataset", required=True)
    parser.add_argument("--output_format", type=str, help="Select between [coco, voc]", required=True)
    parser.add_argument("--output_dir", type=str, help="Folder to save the annotations.", default="./")

    args = parser.parse_args()
    converter(args)
