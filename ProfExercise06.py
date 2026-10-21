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

###############################################################################
# MAIN
###############################################################################

def main(): 
    '''   
    complex_data = fh.make_simple_complex()
    mag, phase = fh.complex_to_polar(complex_data)
    complex_image = fh.to_complex_image(complex_data)
    
    cv2.normalize(complex_image, complex_image, 
                    norm_type=cv2.NORM_MINMAX)
    cv2.normalize(mag, mag, 
                    norm_type=cv2.NORM_MINMAX)
    cv2.normalize(phase, phase, 
                    norm_type=cv2.NORM_MINMAX)
    
    cv2.imshow("Original X", complex_image[...,0])
    cv2.imshow("Original Y", complex_image[...,1])
    cv2.imshow("Magnitude", mag)
    cv2.imshow("Phase", phase)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    '''
    
    simple1D = np.array([[1,2,4,4]])
    simple_fourier = fh.compute_fourier(simple1D)
    simple_complex = fh.to_numpy_complex(simple_fourier)
    print("Original:", simple1D)
    print("Fourier:", simple_complex)
    
    simple_mag, simple_phase = fh.complex_to_polar(simple_complex)
    print("Magnitude:", simple_mag)
    print("Phase:", simple_phase)
    
    simple_recon = fh.compute_inverse_fourier(simple_fourier)
    print("Recon:", simple_recon)
        
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
        
        # Show the image
        cv2.imshow("Video", frame)

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
    