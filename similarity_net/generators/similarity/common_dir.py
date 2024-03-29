import keras
import os
import random
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

from .utils import resize_image, get_img_resize_scale
from .similarity_generator import SimilarityGenerator
from ..common_dir_utils import dir_dict_from_file, get_class_contained_images

class PairDescription:
    def __init__(self, pair_matches, first_image_path, second_image_path):
        self.pair_matches = pair_matches
        self.first_image_path = first_image_path
        self.second_image_path = second_image_path

class CommonDirSimilarityGenerator(SimilarityGenerator):
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

        super(CommonDirSimilarityGenerator, self).__init__(**kwargs)

    def initialize(self, batch_size, steps_per_epoch, proportion_matching):
        # print("Initializing with steps {}, proportion {}, and batch size {}".format(steps_per_epoch, proportion_matching, batch_size))
        class_names = list(self.class_contained_images.keys())
        self.batch_pair_descriptions = []

        for i in range(steps_per_epoch):
            pair_descriptions = []

            for j in range(batch_size):
                pair_matches = random.uniform(0, 1) < proportion_matching
                # print("Pair matches: ", pair_matches)

                first_image_class = random.choice(class_names)

                if pair_matches:
                    second_image_class = first_image_class
                else:
                    possible_second_classes = [name for name in class_names if name != first_image_class]
                    second_image_class = random.choice(possible_second_classes)

                first_image_filename = random.choice(self.class_contained_images[first_image_class])
                first_image_path = os.path.join(self.class_dir_dict[first_image_class], first_image_filename)
                first_image_path = os.path.join(self.root_path, first_image_path)

                second_image_filename = random.choice(self.class_contained_images[second_image_class])
                second_image_path = os.path.join(self.class_dir_dict[second_image_class], second_image_filename)
                second_image_path  = os.path.join(self.root_path, second_image_path)

                pair_description = PairDescription(pair_matches, first_image_path, second_image_path)
                pair_descriptions.append(pair_description)

            self.batch_pair_descriptions.append(pair_descriptions)

        random.shuffle(self.batch_pair_descriptions)

    @staticmethod
    def load_image_from_path(image_path, min_img_size=800, max_img_size=1400):
        pil_img = load_img(image_path)
        img = img_to_array(pil_img)
        # img = load_image_as_np_array(image_path)
        scale = get_img_resize_scale(img.shape, min_img_size=min_img_size, max_img_size=max_img_size)

        img = resize_image(img, scale)

        # img /= 127.5
        # img -= 1.

        return img

    def get_input_output(self, index):
        pair_descriptions = self.batch_pair_descriptions[index]

        first_images = []
        second_images = []
        scores = []

        for description in pair_descriptions:
            if description.pair_matches:
                score = 1.0
            else:
                score = 0.0

            first_image = self.load_image_from_path(description.first_image_path)
            second_image = self.load_image_from_path(description.second_image_path)

            # inputs.append([first_image, second_image])
            first_images.append(first_image)
            second_images.append(second_image)
            scores.append(score)

        inputs = [np.array(first_images), np.array(second_images)]
        outputs = np.array(scores)

        return inputs, outputs
