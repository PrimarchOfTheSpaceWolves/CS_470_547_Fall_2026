import numpy as np
import cv2

def to_numpy_complex(complex_image):
    return (complex_image[:,:,0] + complex_image[:,:,1]*1j)

def to_complex_image(complex_data):
    return np.stack([
            np.real(complex_data),
            np.imag(complex_data)
        ], axis=2)
    
def complex_to_polar(complex_data):
    mag = np.abs(complex_data)
    phase = np.angle(complex_data)
    return mag, phase
    
def make_simple_complex(length=600):
    hl = length/2
    complex_data = np.zeros((length, length), dtype="complex")
    
    values = np.arange(-hl, hl, 1)
    
    complex_data[:] = values
    complex_data += np.reshape(1j*values, (-1,1))
    
    return complex_data
