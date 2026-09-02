import cv2
import numpy as np

def main():
    
    gradient = np.zeros((480,640), dtype="uint8")
    
    key = -1
    ESC_KEY = 27
    
    while key != ESC_KEY:
        
        cv2.imshow("Gradient", gradient)
        key = cv2.waitKey(33)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    