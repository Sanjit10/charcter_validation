from scripts.skeletonize import ImageProcessor
import os
import glob
import cv2 as cv
import time

extensions = ['jpg', 'png', 'bmp']

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
                                        skeletonize=True,
                                        erode=True,
                                        dilate=False,
                                        iterations=1
                        )
        print("Time taken for morphological processing:", time.time()-st)
        cv.imshow("image",cv.imread(path))
        cv.imshow("binary_image", binary_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()