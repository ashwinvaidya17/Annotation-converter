import argparse
from converters import convert


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert data from Unity annotation to selected annotation format.")
    parser.add_argument("--input_dir", type=str, help="Path to dataset", required=True)
    parser.add_argument(
        "--input_format",
        type=str,
        help="Format of the input dataset. Supported Formats are ['egocentric_food', 'unity_perception']",
        default="unity_perception",
    )
    parser.add_argument(
        "--output_format", type=str, help="Select between [coco, voc, simplified_detection]", required=True
    )
    parser.add_argument("--output_dir", type=str, help="Folder to save the annotations.", default="./")

    args = parser.parse_args()
    convert(args)
