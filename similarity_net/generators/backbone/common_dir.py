from .backbone_generator import BackboneGenerator
from ..common_dir_utils import dir_dict_from_file, get_class_contained_images
from keras.preprocessing.image import img_to_array, load_img

class CommonDirBackboneGenerator(BackboneGenerator):
    def __init__(
        self,
        dir_list_filepath,
        root_path,
        **kwargs):

        self.class_name_indices = {}
        for i, class_name in enumerate(self.class_dir_dict.keys()):
            self.class_name_indices[class_name] = i

        self.class_dir_dict = dir_dict_from_file(dir_list_filepath)

        self.root_path = root_path

        self.class_contained_images = get_class_contained_images(self.class_dir_dict, self.root_path)

        self.class_filename_pairs = []
        for class_name in self.class_contained_images:
            for filename in self.class_contained_images[class_name]:
                self.class_filename_pairs.append((class_name, filename))

        super(CommonDirBackboneGenerator, self).__init__(**kwargs)

    def num_items(self):
        """ Number of training items available
        """
        return len(self.class_filename_pairs)

    def load_item(self, idx):
        """ Returns image np array along with label integer
        """
        class_name, filename = self.class_filename_pairs[idx]
        class_idx = self.class_name_indices[class_name]
        class_dir = os.path.join(self.root_path, self.class_dir_dict[class_name])
        filepath = os.path.join(class_dir, filename)

        img = img_to_array(load_img(image_path))
        print("img shape: ", img.shape)
        print("class: ", class_idx)
        return img, class_idx
        # raise NotImplementedError()




