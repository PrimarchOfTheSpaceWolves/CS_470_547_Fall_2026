import cv2
import numpy as np

def main():
    
    gradient = np.zeros((480,640), dtype="uint8")
    float_gradient = np.zeros(gradient.shape, dtype="float64")
    
    test_contrast = np.zeros((800,800), dtype="uint8")
    back_gray = 0
    fore_gray = 255
    
    key = -1
    ESC_KEY = 27
    
    max_gray = 100
    
    while key != ESC_KEY:
        test_contrast[:,:] = back_gray
        test_contrast[200:600,100:700] = fore_gray        
        
        for i in range(gradient.shape[1]):
            fraction = i / (gradient.shape[1]-1)
            float_gradient[:,i:(i+1)] = fraction*max_gray
            
        gradient = cv2.convertScaleAbs(float_gradient)
                    
        cv2.imshow("Gradient", gradient)
        cv2.imshow("Float Gradient", float_gradient/255.0)
        cv2.imshow("Contrast", test_contrast)
        
        key = cv2.waitKey(33)
        
        if key == ord('w'): fore_gray = np.clip(fore_gray+1, 0, 255)
        if key == ord('s'): fore_gray = np.clip(fore_gray-1, 0, 255)
                
        if key == ord('e'): back_gray = np.clip(back_gray+1, 0, 255)
        if key == ord('d'): back_gray = np.clip(back_gray-1, 0, 255)
        
        print(fore_gray, back_gray)
        
        if key == ord('q'): max_gray += 5
        if key == ord('a'): max_gray -= 5
        #print(max_gray)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    