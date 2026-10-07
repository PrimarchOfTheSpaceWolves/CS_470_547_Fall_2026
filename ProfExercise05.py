###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import cv2
import pandas
import sklearn
import timm
import torchvision
from enum import Enum

class FilterType(Enum):
    BOX = "Box filter"
    GAUSS = "Gaussian filter"
    MEDIAN = "Median filter"
    
def do_filter(image, filter_size, filter_type):
    if filter_type == FilterType.BOX:
        output = cv2.blur(image, (filter_size, filter_size))    
    elif filter_type == FilterType.GAUSS:
        output = cv2.GaussianBlur(image, 
                                  ksize=(filter_size,filter_size),
                                  sigmaX=0)
    elif filter_type == FilterType.MEDIAN:
        output = cv2.medianBlur(image, filter_size)
       
    return output

def do_add_salt_pepper_noise(image, prob):
    # Salt
    choice = np.random.rand(image.shape[0], image.shape[1])    
    output = np.where(choice < prob, 255, image)    
    # Pepper
    choice = np.random.rand(image.shape[0], image.shape[1])    
    output = np.where(choice < prob, 0, output)  
    return output 

def do_add_noise(image, scale):
    fimage = image.astype("float32")
    
    choice = np.random.rand(image.shape[0], image.shape[1])
    choice = 2.0*choice - 1.0
    choice = scale*choice    
    fimage = fimage + choice  
    
    output = np.clip(np.round(fimage), 0, 255).astype("uint8") 
     
    return output 
    
###############################################################################
# MAIN
###############################################################################

def main():        
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    b = torch.rand(5,3)
    print("Random Torch Numbers:")
    print(b)
    print("Do you have Torch CUDA/ROCm?:", torch.cuda.is_available())
    print("Do you have Torch MPS?:", torch.mps.is_available())
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Torch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("timm:", timm.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
        
    print("FILTERING OPTIONS:")
    for index, item in enumerate(list(FilterType)):
        print(index, "-", item.value)
    filter_type = list(FilterType)[int(input("Enter choice: "))] 
    filter_size = 3
            
    ###############################################################################
    # OPENCV
    ###############################################################################
    
    # Opening video...
    print("Opening the video...")
    video_file = "images/noice.mp4"
    capture = cv2.VideoCapture(video_file) 
                
    # Did we get it?
    if not capture.isOpened():
        print("ERROR: Cannot open the video: %s" % video_file)
        exit(1)
        
    add_salt_pepper_noise = False
    noise_scale = 0.0

    # While not the escape key...
    ESC_KEY = 27
    key = -1
    while key != ESC_KEY:
        # Get next frame from video
        _, frame = capture.read()
        
        # Loop video
        frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if(frame_cnt != -1 and frame_index == frame_cnt):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        grayscale = do_add_noise(grayscale, noise_scale)
        
        if add_salt_pepper_noise:
            grayscale = do_add_salt_pepper_noise(grayscale, 0.01)
               
        output = do_filter(grayscale, filter_size, filter_type)
        
        # Show the image
        cv2.imshow("Video", grayscale)
        cv2.imshow("Filtered", output)

        # Wait 30 milliseconds, and grab any key presses
        key = cv2.waitKey(30)
        
        if key == ord('q'):
            filter_size += 2
            print("Size:", filter_size)
        if key == ord('a'):
            filter_size = max(3, filter_size-2)
            print("Size:", filter_size)
            
        if key == ord('z'):
            add_salt_pepper_noise = not add_salt_pepper_noise
            
        if key == ord('w'):
            noise_scale += 1.0
        if key == ord('s'):
            noise_scale = max(0.0, noise_scale - 1.0)
            
    # Release the capture and destroy the window
    capture.release()
    cv2.destroyAllWindows()

    # Close down...
    print("Closing application...")


# The main function
if __name__ == "__main__": 
    main()
    