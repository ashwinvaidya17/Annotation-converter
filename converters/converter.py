from converters import unity_perception_converters, egocentric_food_converters


def unity_converter(args) -> None:

    output_format = args.output_format
    if output_format == "coco":
        writer = unity_perception_converters.COCOWriter(args.input_dir, args.output_dir)
    elif output_format == "simplified_detection":
        writer = unity_perception_converters.SimplifiedDetectionWriter(args.input_dir, args.output_dir)
    elif output_format == "voc":
        writer = unity_perception_converters.VOCWriter(args.input_dir, args.output_dir)
    else:
        raise NotImplementedError(f"Output format {output_format} not supported yet.")

    writer.write()


def egocentric_food_converter(args) -> None:
    output_format = args.output_format
    if output_format == "voc":
        writer = egocentric_food_converters.VOCWriter(args.input_dir, args.output_dir)
    else:
        raise NotImplementedError(f"Output format {output_format} not supported yet.")

    writer.write()


def convert(args):
    if args.input_format == "unity_perception":
        unity_converter(args)
    elif args.input_format == "egocentric_food":
        egocentric_food_converter(args)
    else:
        raise NotImplementedError(f"Input format {args.input_format} not supported yet.")
