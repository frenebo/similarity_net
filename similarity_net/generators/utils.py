from PIL import Image
import numpy as np
import cv2

# def load_image_as_np_array(image_path):
#     return np.asarray(Image.open(image_path).convert("RGB"))

def get_img_resize_scale(orig_np_shape, min_img_size, max_img_size):
    (orig_height, orig_width, _) = orig_np_shape

    smallest_side = min(orig_height, orig_width)
    largest_side = max(orig_height, orig_width)

    scale = smallest_side / min_img_size
    if largest_side * scale > max_img_size:
        scale = max_img_side / largest_side

    return scale

def resize_image(np_img, scale):
    resized_img = cv2.resize(np_img, None, fx=scale, fy=scale)

    return resized_img
