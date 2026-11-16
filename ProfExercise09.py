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
import math as m

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
        
    # Set initial thresholds
    lowT = 100
    highT = 200

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
        edges = cv2.Canny(grayscale, lowT, highT)
        
        lineImage = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        lineSegments = cv2.HoughLinesP(
            edges, 1.0, m.pi/180.0, 50, 
            minLineLength=10.0, 
            maxLineGap=10.0
        )
        
        #print(lineSegments.shape)
        for line in lineSegments:
            cv2.line(lineImage, 
                     (line[0], line[1]),
                     (line[2], line[3]),
                     (0,0,255), 3)
        
        # Show the image
        cv2.imshow("Video", grayscale)
        cv2.imshow("Edges", edges)
        cv2.imshow("Lines", lineImage)

        # Wait 30 milliseconds, and grab any key presses
        key = cv2.waitKey(30)
        
        if key == ord('q'): lowT += 10
        if key == ord('a'): lowT -= 10
        if key == ord('w'): highT += 10
        if key == ord('s'): highT -= 10
        print("Thresh:", lowT, highT)

    # Release the capture and destroy the window
    capture.release()
    cv2.destroyAllWindows()

    # Close down...
    print("Closing application...")


# The main function
if __name__ == "__main__": 
    main()
    