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

class ThresholdType(Enum):
    BINARY = "Binary Threshold"
    ESTIMATED = "Estimated Threshold"
    OTSU = "Otsu's Method"
    
def threshold(image, thresh_type, user_value=128):
    if thresh_type == ThresholdType.BINARY:
        _, output = cv2.threshold(image, user_value, 255, cv2.THRESH_BINARY)
    elif thresh_type == ThresholdType.ESTIMATED:
        T = cv2.mean(image)[0]
        Tdiff = 1000
        MIN_DIFF = 1
        while Tdiff > MIN_DIFF:
            _, output = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
            fore_mask = output
            back_mask = 255 - output
            Tf = cv2.mean(image, fore_mask)[0]
            Tb = cv2.mean(image, back_mask)[0]
            Told = T
            T = (Tf + Tb)/2
            Tdiff = abs(T - Told)
           
        _, output = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)  
        print("Threshold:", T)   
    elif thresh_type == ThresholdType.OTSU:
        chosen, output = cv2.threshold(image, user_value, 255, 
                                       cv2.THRESH_OTSU)     
        print("Threshold:", chosen)           
                
    return output

###############################################################################
# MAIN
###############################################################################

def main():    
    
    chosenT = 0
    print("THRESHOLD APPROACH:")
    for index, item in enumerate(list(ThresholdType)):
        print(index, "-", item.value)
    chosenT = list(ThresholdType)[int(input("Enter choice: "))] 
            
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

    user_thresh = 128
    
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
        output = threshold(grayscale, chosenT, user_thresh)
        
        # Show the image
        cv2.imshow("Video", grayscale)
        cv2.imshow("Threshold", output)

        # Wait 30 milliseconds, and grab any key presses
        key = cv2.waitKey(30)
        
        if key == ord("q"): user_thresh += 4
        if key == ord("a"): user_thresh -= 4
        user_thresh = max(user_thresh, 3)

    # Release the capture and destroy the window
    capture.release()
    cv2.destroyAllWindows()

    # Close down...
    print("Closing application...")


# The main function
if __name__ == "__main__": 
    main()
    