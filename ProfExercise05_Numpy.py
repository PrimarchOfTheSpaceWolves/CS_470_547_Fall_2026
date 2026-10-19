import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def main():
    height = 4
    width = 5
    size = height*width
    
    window_size = (3,4)
    
    image = np.arange(size).reshape(height, width)
    print("Image shape:", image.shape)
    print("Image:\n", image)
    
    patches = sliding_window_view(image, window_size)
    print("Patches shape:", patches.shape)
    print("Patches:\n", patches) 
    
    kernel = np.zeros(window_size)
    kernel[:,:] = 1.0/(window_size[0]*window_size[1])
    
    flipped_kernel = kernel[::-1,::-1]
    
    output = np.tensordot(patches, flipped_kernel, axes=([2,3],[0,1]))
    print("Output shape:", output.shape)
    print("Output:\n", output)
    
if __name__ == "__main__":
    main()
    
