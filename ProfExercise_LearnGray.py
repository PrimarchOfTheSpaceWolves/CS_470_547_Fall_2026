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
import matplotlib.pyplot as plt
from enum import Enum
from torch import nn
from torchvision.transforms import v2

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
    
    conv_layer = nn.Conv2d(3, 1, 1, bias=False)
    model = nn.Sequential(conv_layer)
    print(model)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    device = "cuda" # "cpu"
    model = model.to(device)
    
    data_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
            
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
        # Webcam
        print("Opening the webcam...")

        # Linux/Mac (or native Windows) with direct webcam connection
        capture = cv2.VideoCapture("images/noice.mp4")
        # Did we get it?
        if not capture.isOpened():
            print("ERROR: Cannot open the camera!")
            exit(1)

        # While not closed...
        key = -1
        ESC_KEY = 27
        while key != ESC_KEY:
            # Get next frame from camera
            _, image = capture.read()
            
            frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_cnt != -1 and frame_cnt == frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            desired_output = np.expand_dims(grayscale, axis=-1)
            desired_output = data_transform(desired_output)
            desired_output = torch.unsqueeze(desired_output, 0)
            
            data_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data_input = data_transform(data_input)
            data_input = torch.unsqueeze(data_input, 0)
            
            # print(data_input.shape, desired_output.shape)
            
            # Show the image
            cv2.imshow("Original", image)
            cv2.imshow("Ground", grayscale)                           
            
            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(15)
            
        # Release the camera and destroy the window
        capture.release()
        cv2.destroyAllWindows()
        
        # Close down...
        print("Closing application...")

    else:
        # Trying to load image from argument

        # Get filename
        filename = sys.argv[1]

        # Load image
        print("Loading image:", filename)
        image = cv2.imread(filename) 
        
        # Check if data is invalid
        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        
        key = -1
        while key == -1:
            cv2.imshow(windowTitle, image)

            # Wait for a keystroke to close the window
            key = cv2.waitKey(30)

        # Cleanup this window
        cv2.destroyAllWindows()

# The main function
if __name__ == "__main__": 
    main()
    