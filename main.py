from scripts.skeletonize import ImageProcessor
import os
import glob
import cv2 as cv
import time
import numpy as np

extensions = ['jpg', 'png', 'bmp']

# Define the callback function to display mouse coordinates
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        # Create a copy of the image to draw coordinates on
        img_with_coords = param.copy()
        cv.putText(img_with_coords, f'X: {x}, Y: {y}', (10, img_with_coords.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Display the image with coordinates
        cv.imshow("bin_image", img_with_coords)

def get_image_files(directory, extensions):
    for ext in extensions:
        for filename in glob.iglob(os.path.join(directory, f'*.{ext}'), recursive=True):
            yield filename

def main():
    
    image_preprocessor = ImageProcessor()
    for path in get_image_files('./debug_inputs', extensions):
        st = time.time()
        binary_image =image_preprocessor.process_image(
                                        image_input=path, 
                                        image_class='A',
                                        skeletonize=True,
                                        erode=True,
                                        dilate=False,
                                        iterations=2,
                                        output_path=r"D:\crimson_tech\charcter_validation\output\output.csv"
                        )
        print("Time taken for morphological processing:", time.time()-st)
        
        # Set up the window and callback function
        cv.namedWindow("bin_image")
        cv.setMouseCallback("bin_image", mouse_callback, param=binary_image)

        cv.imshow("image",cv.imread(path))
        cv.imshow("bin_image", binary_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()