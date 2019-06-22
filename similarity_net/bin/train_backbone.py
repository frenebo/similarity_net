import sys
import os
import argparse
import keras

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import similarity_net.bin
    __package__ = "similarity_net.bin"

from ..generators.similarity.common_dir import CommonDirSimilarityGenerator
from ..models import load_model, create_similaritynet
from ..models.backbones import get_backbone_class

def check_parsed_args(args):
    pass


def parse_command_line_args(command_line_args):
    parser = argparse.ArgumentParser(description="Training script for training Similarity Network")

    parser.add_argument("--from-weights", help="File to load backbone weights from")
    parser.add_argument("--snapshot-path", help="Path to save snapshots")


    parser.add_argument("--epochs", help="How many epochs to train for", type=int, default=10)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=1)
    parser.add_argument("--steps-per-epoch", help="Steps per epoch", type=int, default=10)
    parser.add_argument("--lr", help="Learning rate", default=0.0001, type=float)

    parser.add_argument("backbone", help="Name of backbone to use")

    dataset_subparsers = parser.add_subparsers(help="Arguments for specific dataset types", dest="dataset_type")
    dataset_subparsers.required = True

    common_dir_parser = dataset_subparsers.add_parser("common-dir")
    common_dir_parser.add_argument("dir_list_file", help="File that lists the paths of all dirs")
    common_dir_parser.add_argument("root_dir", help="Root dir from which to look for the dirs listed in dir list file")

    args = parser.parse_args(command_line_args)
    check_parsed_args(args)
    return args

def main():
    args = parse_command_line_args(sys.argv[1:])
    print(args)

if __name__ == "__main__":
    main()