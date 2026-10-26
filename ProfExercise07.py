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
import ProfFreqHelper as fh

def do_frequency(image, mask_image):
    pad_image = fh.pad_for_fourier(image)
    fourier_image = fh.compute_fourier(pad_image)
    
    complex_data = fh.to_numpy_complex(fourier_image)
    mag, phase = fh.complex_to_polar(complex_data)
    
    #print("Mag range:", np.amin(mag), np.amax(mag))
    
    log_mag = cv2.log(mag + 1.0)
    
    display_mag = cv2.normalize(log_mag, dst=None, alpha=0, beta=1, 
                                norm_type=cv2.NORM_MINMAX)
    display_phase = cv2.normalize(phase, dst=None, alpha=0, beta=1,
                                  norm_type=cv2.NORM_MINMAX)
    
    cv2.imshow("Magnitude", display_mag)
    cv2.imshow("Phase", display_phase)    
    
    output = fh.compute_inverse_fourier(fourier_image)
    output = cv2.convertScaleAbs(output)
        
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
        
    mask_image = None

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
        
        if mask_image is None:
            mask_image = np.ones(grayscale.shape, dtype="float64")
        
        output = do_frequency(grayscale, mask_image)
        
        # Show the image
        cv2.imshow("Video", grayscale)
        cv2.imshow("Output", output)

        # Wait 30 milliseconds, and grab any key presses
        key = cv2.waitKey(30)

    # Release the capture and destroy the window
    capture.release()
    cv2.destroyAllWindows()

    # Close down...
    print("Closing application...")


# The main function
if __name__ == "__main__": 
    main()
    