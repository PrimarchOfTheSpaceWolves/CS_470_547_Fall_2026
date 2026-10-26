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

def polar_to_complex(mag, phase):
    return mag * np.exp(1j * phase)
    
def make_simple_complex(length=600):
    hl = length/2
    complex_data = np.zeros((length, length), dtype="complex")
    
    values = np.arange(-hl, hl, 1)
    
    complex_data[:] = values
    complex_data += np.reshape(1j*values, (-1,1))
    
    return complex_data

def compute_fourier(image):
    fimage = image.astype("float64")
    output = cv2.dft(fimage, 
                     flags=cv2.DFT_COMPLEX_OUTPUT)
    return output

def compute_inverse_fourier(freq):
    return cv2.idft(freq, 
                    flags=(cv2.DFT_REAL_OUTPUT
                           +cv2.DFT_SCALE))
    
def pad_for_fourier(image):
    m = cv2.getOptimalDFTSize(image.shape[0])
    n = cv2.getOptimalDFTSize(image.shape[1])
    pad_image = cv2.copyMakeBorder(
                    image,
                    0, m - image.shape[0],
                    0, n - image.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=0)
    return pad_image
