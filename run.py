import argparse
from converters.simplified_detection_writer import SimplifiedDetectionWriter


from converters.coco_writer import COCOWriter


def converter(args):

    output_format = args.output_format
    if output_format == "coco":
        writer = COCOWriter(args.input_dir, args.output_dir)
    elif output_format == "simplified_detection":
        writer = SimplifiedDetectionWriter(args.input_dir, args.output_dir)
    else:
        raise NotImplementedError(f"Output format {output_format} not supported yet")

    writer.write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert data from Unity annotation to selected annotation format.")
    parser.add_argument("--input_dir", type=str, help="Path to unity dataset", required=True)
    parser.add_argument(
        "--output_format", type=str, help="Select between [coco, voc, simplified_detection]", required=True
    )
    parser.add_argument("--output_dir", type=str, help="Folder to save the annotations.", default="./")

    args = parser.parse_args()
    converter(args)
