import sys
import os
import argparse

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import similarity_net.bin  # noqa: F401
    __package__ = "similarity_net.bin"

from ..generators.common_dir_generator import CommonDirGenerator

def check_parsed_args(args):
    if args.snapshot_path is None and not args.no_snapshots:
        raise ValueError("Must provide --snapshot-path PATH or specify --no-snapshots")

def parse_command_line_args(command_line_args):
    parser = argparse.ArgumentParser(description="Training script for training Similarity Network")

    parser.add_argument("--backbone-weights", help="File to load backbone weights from")
    parser.add_argument("--gpu-names", help="Specify names of GPUs that should be visible")
    parser.add_argument("--snapshot-path", help="Path to save snapshots")
    parser.add_argument("--no-snapshots", help="Flag to not use snapshots", action="store_true")

    parser.add_argument("--epochs", help="How many epochs to train for", type=int, default=10)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=1)
    parser.add_argument("--steps-per-epoch", help="Steps per epoch", type=int, default=10)
    parser.add_argument("--proportion-matching", help="Proportion of matching images that the generator should produce", type=float, default=0.5)

    # subparsers = parser.add_argument(h)
    # parser.add_argum
    subparsers = parser.add_subparsers(help="Arguments for specific dataset types", dest="dataset_type")
    subparsers.required = True

    common_dir_parser = subparsers.add_parser("common-dir")
    common_dir_parser.add_argument("dir_list_file", help="File that lists the paths of all dirs")
    common_dir_parser.add_argument("root_dir", help="Root dir from which to look for the dirs listed in dir list file")

    args = parser.parse_args(command_line_args)
    check_parsed_args(args)
    return args

def main():
    args = parse_command_line_args(sys.argv[1:])

    if args.gpu_names:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_names

    if args.dataset_type == "common-dir":
        generator = CommonDirGenerator(
            dir_list_filepath=args.dir_list_file,
            root_path=args.root_dir,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            proportion_matching=args.proportion_matching,
        )
    else:
        raise ValueError("Unimplemented dataset type '{}'".format(args.dataset_type))

    print(args)

if __name__ == "__main__":
    main()
