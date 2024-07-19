from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, erosion, dilation
import numpy as np
from skan import Skeleton, summarize
import pandas as pd
from processor_factory import ProcessorFactory

class ImageProcessor:
    
    def __init__(self):
        self.branch_data_df = pd.DataFrame()
        self.count = 0
        self.processor = ProcessorFactory()
        self.result = "Error"

    def load_image(self, image_input: str|np.ndarray ) -> np.ndarray:
        """
            Loads an image from the given path.
            Input:
                image_input (str|np.ndarray): Path to the image or the image itself.
            Output:
               image (np.ndarray): The loaded image.
            
        """
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
        
        self.image = image
        threshold_value = threshold_otsu(image)
        binary_image = image > threshold_value
        self.binary_image = binary_image
        return binary_image

    def skeletonize_image(self, binary_image: np.ndarray) -> np.ndarray:
        """
            Skeletonizes the given image.
            Input:
                binary_image (np.ndarray): The image to be skeletonized.
            Output:
                skeletonized (np.ndarray): The skeletonized image.
        """
        return skeletonize(binary_image)

    def erode_image(self, binary_image: np.ndarray, iterations:int=1) -> np.ndarray:
        """
            Erodes the image passed to the funciton.
            Input:
                binary_image (np.ndarray): The image to be eroded.
                iteration (default=1): The number of iterations to be performed on the image.
            Output:
                eroded (np.ndarray): The eroded image.
        """
        structuring_element = np.ones((3, 3), dtype=bool)
        eroded = binary_image
        for _ in range(iterations):
            eroded = erosion(eroded, structuring_element)
        return eroded

    def dilate_image(self, binary_image: np.ndarray, iterations:int=1) -> np.ndarray:
        """
            Dilates the image passed to the function.
            Input:
                binary_image (np.ndarry): The image to be dialated
                iterations (int): The number of iterations to be performed on the image.
            Output:
                dilated (np.ndarray): The dilated image.
        """
        structuring_element = np.ones((3, 3), dtype=bool)
        dilated = binary_image
        for _ in range(iterations):
            dilated = dilation(dilated, structuring_element)
        return dilated

    def find_branching_and_end_points(self, skeleton: np.ndarray, output_path: str, image_class: str) -> None:
        """
            Processes the skeleton to find its end point and other metadata.
            Input:
                skeleton (np.ndarray): The skeleton image to be processed
                output_path (str): The output path to save the metadata
                image_class (str): Class of processor to be produced from factory/ also the class of image in detection
            Output:
                None
        """
        
        
        self.count +=1  
        branch_data = summarize(Skeleton(skeleton))
        self.process_skeleton_metadata(branch_data, image_class)
        
        branch_data['skeleton-id']=self.count
         # Append the new data to the existing DataFrame
        self.branch_data_df = pd.concat([self.branch_data_df, branch_data])
        
        print(self.branch_data_df)
        # Save the updated DataFrame to a CSV file
        self.branch_data_df.to_csv(output_path, index=False)
    
    # Character Specific Processing
    def process_skeleton_metadata(self, data:pd.DataFrame, image_class:str):
        """
            Function that generates a class specific character function from the factory.
            Input:
                data(Dataframe): The dataframe containing the skeleton metadata
                image_class (str): Class of image in detection / class of processor
            Output:
                None (updates self.result so no need to return result)
        """
        processor = self.processor.get_processor(str(image_class))
        result = processor.process(data)
        self.result = result
    


    def process_image(self, image_input, image_claass='A', skeletonize=True, erode=False, dilate=False, iterations=1, output_path = "./"):
        """
            Main Entry point of the class, acts as pipeline calling different functions serially as per
            the params passed
            Input:
                image_input (str | np.ndarray): The image to be processed
                image_class (str = 'A'): The class of object detection/ class of factory
                erode (bool = Fasle): To erode before skeletanization or not 
                dialate (bool = False): To dilate before skeletanization or not
                iteration (int = 1): No of iterations the image is eroded or dialated
                output_path(str = "./"): Path to output the details of the skeleton (as csv)
                
        """
        binary_image = self.load_image(image_input)

        if erode:
            binary_image = self.erode_image(binary_image, iterations)
        if dilate:
            binary_image = self.dilate_image(binary_image, iterations)
        if skeletonize:
            binary_image = self.skeletonize_image(binary_image)
            self.find_branching_and_end_points(binary_image, output_path, image_claass)

        binary_image_uint8 = (binary_image * 255).astype(np.uint8)
        return binary_image_uint8, self.result #self.result is directly updated by the process_skeleton_metadata function
