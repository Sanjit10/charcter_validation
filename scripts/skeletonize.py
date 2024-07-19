import cv2
import numpy as np
from skimage.morphology import skeletonize, erosion, dilation

class ImageProcessor:
    def __init__(self):
        pass

    def load_image(self, image_input):
        if isinstance(image_input, str):
            image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Image not found at path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError("Invalid image input type. Expected file path or NumPy array.")
        
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        binary_image = binary_image == 255
        return binary_image

    def skeletonize_image(self, binary_image):
        skeleton = skeletonize(binary_image)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        return skeleton_uint8

    def erode_image(self, binary_image, iterations=1):
        eroded = erosion(binary_image, np.ones((3, 3), dtype=bool))
        for _ in range(iterations - 1):
            eroded = erosion(eroded, np.ones((3, 3), dtype=bool))
        eroded_uint8 = (eroded * 255).astype(np.uint8)
        return eroded_uint8

    def dilate_image(self, binary_image, iterations=1):
        dilated = dilation(binary_image, np.ones((3, 3), dtype=bool))
        for _ in range(iterations - 1):
            dilated = dilation(dilated, np.ones((3, 3), dtype=bool))
        dilated_uint8 = (dilated * 255).astype(np.uint8)
        return dilated_uint8

    def process_image(self, image_input, skeletonize = True, erode = False, dilate = False,  iterations=1):
        
        binary_image = self.load_image(image_input)
        if erode:
            binary_image = self.erode_image(binary_image, iterations)
        if dilate:
            binary_image = self.dilate_image(binary_image, iterations)
        if skeletonize:
            binary_image = self.skeletonize_image(binary_image)
        
        return binary_image
