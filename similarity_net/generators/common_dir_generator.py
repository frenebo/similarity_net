import keras
import csv
import os
import random
import imghdr

from .similarity_generator import SimilarityGenerator

def dir_dict_from_file(dir_list_filepath):
    with open(dir_list_filepath, "r", newline="") as dir_list_file:
        csv_reader = csv.reader(dir_list_file, delimiter=",")

        class_dir_dict = {}

        for line_no, row in enumerate(csv_reader, 1):
            # skip empty lines
            if not row:
                continue

            try:
                class_name, class_dir = row
            except ValueError:
                raise ValueError("Line {}: should be formatted 'class_name,class_dir'".format(line_no))

            class_dir_dict[class_name] = class_dir

    return class_dir_dict

def get_class_contained_images(class_dir_dict, root_path):
    class_contained_images = {}

    for class_name, class_dir in class_dir_dict.items():
        contained_images = []

        full_dir_path = os.path.join(root_path, class_dir)
        for filename in sorted(os.listdir(full_dir_path)):
            # Check if it's an image
            if imghdr.what(os.path.join(full_dir_path, filename)) is None:
                continue

            contained_images.append(filename)

        if len(contained_images) == 0:
            raise ValueError("Couldn't find any images in directory '{}'".format(full_dir_path))

        class_contained_images[class_name] = contained_images

    return class_contained_images

class PairDescription:
    def __init__(self, pair_matches, first_image_path, second_image_path):
        self.pair_matches = pair_matches
        self.first_image_path = first_image_path
        self.second_image_path = second_image_path

class CommonDirGenerator(SimilarityGenerator):
    def __init__(
        self,
        dir_list_filepath,
        root_path,
        **kwargs):
        self.class_dir_dict = dir_dict_from_file(dir_list_filepath)
        self.root_path = root_path

        self.class_contained_images = get_class_contained_images(self.class_dir_dict, self.root_path)

        # Filled in self.initialize
        self.batch_pair_descriptions = None

        super(CommonDirGenerator, self).__init__(**kwargs)

    def initialize(self, batch_size, steps_per_epoch, proportion_matching):
        class_names = list(self.class_contained_images.keys())
        self.batch_pair_descriptions = []

        for i in range(steps_per_epoch):
            pair_descriptions = []

            for j in range(batch_size):
                pair_matches = random.uniform(0, 1) < proportion_matching

                first_image_class = random.choice(class_names)

                if pair_matches:
                    second_image_class = first_image_class
                else:
                    possible_second_classes = [name for name in class_names if name != first_image_class]
                    second_image_class = random.choice(possible_second_classes)

                first_image_path  = random.choice(self.class_contained_images[first_image_class])
                second_image_path = random.choice(self.class_contained_images[second_image_class])

                pair_description = PairDescription(pair_matches, first_image_path, second_image_path)
                pair_descriptions.append(pair_description)

            self.batch_pair_descriptions.append(pair_descriptions)
