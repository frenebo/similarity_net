import sys
import os
import argparse
import keras

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import similarity_net.bin
    __package__ = "similarity_net.bin"

from ..generators.backbone.common_dir import CommonDirBackboneGenerator
from ..models import load_model, create_similaritynet
from ..models.backbones import get_backbone_class
from ..utils import makedirs

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


def create_callbacks(snapshot_path=None):
    callbacks = []

    if snapshot_path is not None:
        # ensure directory created firs
        makedirs(snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                snapshot_path,
                'backbone_snapshot_{epoch:02d}.h5',
            ),
            verbose=1,
        )
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.2,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    return callbacks

def main():
    args = parse_command_line_args(sys.argv[1:])

    if args.dataset_type == "common-dir":
        generator = CommonDirBackboneGenerator(
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        dir_list_filepath=args.dir_list_file,
        root_path=args.root_dir,
    )

    # Create backbone model
    BackboneClass = get_backbone_class(args.backbone)
    if args.from_weights is not None:
        backbone_model = BackboneClass.from_weights(args.from_weights)
    else:
        backbone_model = BackboneClass()

    # Create callbacks
    callbacks = create_callbacks(snapshot_path=args.snapshot_path)

    # Compile model
    backbone_model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.adam(lr=args.lr, clipnorm=0.01)
    )

    # Train model
    backbone_model.fit_generator(
        generator=generator,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

if __name__ == "__main__":
    main()