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

class IntTransform(Enum):
    ORIGINAL = "Original"
    NEGATIVE = "Negative"

def create_transform_plot(transform, title="Transformation"):
    fig, subfig = plt.subplots(1, 1, figsize=(5,5))
    x = np.arange(256)
    line = subfig.plot(x, transform, color="black", linewidth=1)
    fill = subfig.fill_between(x, transform, color="gray", alpha=0.5)
    subfig.set_xlim([0,255])
    subfig.set_ylim([0,255])
    subfig.set_xlabel("Input intensity")
    subfig.set_ylabel("Output intensity")
    subfig.set_title(title)
    return fig, fill, line[0]

def update_transform_plot(transform, fig, fill, line):
    line.set_ydata(transform)
    
    x_coords = np.arange(256)
    x_coords = np.append(x_coords, [255,0])
    y_coords = np.copy(transform)
    y_coords = np.append(y_coords, [0,0])
    fill.set_verts([np.column_stack([x_coords,y_coords])])
    
    fig.canvas.draw()
    fig.canvas.flush_events()    

def do_transform(image, chosenT):
    if chosenT == IntTransform.ORIGINAL:
        output = np.copy(image)
        transform = np.arange(256, dtype="uint8")
    elif chosenT == IntTransform.NEGATIVE:
        output = 255 - image
        transform = np.arange(255, -1, -1, dtype="uint8")
                
    return output, transform

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
    
    
    chosenT = 0
    print("INTENSITY TRANSFORMS:")
    for index, item in enumerate(list(IntTransform)):
        print(index, "-", item.value)
    chosenT = list(IntTransform)[int(input("Enter choice: "))] 
    
    
    plt.ion()
    fig, fill, line = create_transform_plot(np.arange(256, dtype="uint8"))
        
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

        # Create window ahead of time
        windowName = "Webcam"
        cv2.namedWindow(windowName)

        # While not closed...
        key = -1
        while key == -1:
            # Get next frame from camera
            _, image = capture.read()
            
            frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_cnt != -1 and frame_cnt == frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output, transform = do_transform(grayscale, chosenT)
    
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("Transformed", output)
            update_transform_plot(transform, fig, fill, line)           
            
            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)

        # Release the camera and destroy the window
        capture.release()
        cv2.destroyAllWindows()
        plt.close()

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
    