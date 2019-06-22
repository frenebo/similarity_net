import imghdr
import os
import csv
from collections import OrderedDict

def dir_dict_from_file(dir_list_filepath):
    with open(dir_list_filepath, "r", newline="") as dir_list_file:
        csv_reader = csv.reader(dir_list_file, delimiter=",")

        class_dir_dict = OrderedDict()

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
