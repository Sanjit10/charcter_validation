from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, erosion, dilation
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

    def load_image(self, image_input):
        if isinstance(image_input, str):
            image = io.imread(image_input, as_gray=True)
            if image is None:
                raise ValueError(f"Image not found at path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                image = color.rgb2gray(image_input)
            else:
                image = image_input
        else:
            raise ValueError("Invalid image input type. Expected file path or NumPy array.")
        
        threshold_value = threshold_otsu(image)
        binary_image = image > threshold_value
        return binary_image

    def skeletonize_image(self, binary_image):
        return skeletonize(binary_image)

    def erode_image(self, binary_image, iterations=1):
        structuring_element = np.ones((3, 3), dtype=bool)
        eroded = binary_image
        for _ in range(iterations):
            eroded = erosion(eroded, structuring_element)
        return eroded

    def dilate_image(self, binary_image, iterations=1):
        structuring_element = np.ones((3, 3), dtype=bool)
        dilated = binary_image
        for _ in range(iterations):
            dilated = dilation(dilated, structuring_element)
        return dilated

    def process_image(self, image_input, skeletonize=True, erode=False, dilate=False, iterations=1):
        binary_image = self.load_image(image_input)

        if erode:
            binary_image = self.erode_image(binary_image, iterations)
        if dilate:
            binary_image = self.dilate_image(binary_image, iterations)
        if skeletonize:
            binary_image = self.skeletonize_image(binary_image)

        binary_image_uint8 = (binary_image * 255).astype(np.uint8)
        return binary_image_uint8
