import cv2
import numpy as np

def main():
    
    gradient = np.zeros((480,640), dtype="uint8")
    
    key = -1
    ESC_KEY = 27
    
    max_gray = 100
    
    while key != ESC_KEY:
        for i in range(gradient.shape[1]):
            fraction = i / (gradient.shape[1]-1)
            gradient[:,i:(i+1)] = fraction*max_gray
                    
        cv2.imshow("Gradient", gradient)
        key = cv2.waitKey(33)
        
        if key == ord('q'): max_gray += 5
        if key == ord('a'): max_gray -= 5
        print(max_gray)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    