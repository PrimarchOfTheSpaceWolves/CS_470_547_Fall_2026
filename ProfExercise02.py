import cv2
import numpy as np

def gray_slice(image, min_val, max_val):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.where(image < min_val, min_val, image)
    image = np.where(image > max_val, max_val, image)
    return image

def main():
    

    
    gradient = np.zeros((480,640), dtype="uint8")
    float_gradient = np.zeros(gradient.shape, dtype="float64")
    
    test_contrast = np.zeros((800,800), dtype="uint8")
    back_gray = 0
    fore_gray = 255
    
    key = -1
    ESC_KEY = 27
    
    max_gray = 100
    
    capture = cv2.VideoCapture("images/noice.mp4")
    
    if not capture.isOpened():
        print("Could not open video!")
        exit(1)
    
    min_slice = 100
    max_slice = 200
    
    scale = 1
    
    while key != ESC_KEY:
        
        _, frame = capture.read()
        
        resized_frame = cv2.resize(frame, dsize=None, 
                                   fx=1.0/scale, fy=1.0/scale,
                                   interpolation=cv2.INTER_LINEAR)
        resized_frame = cv2.resize(resized_frame, dsize=None,
                                   fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)
        cv2.imshow("MY EYES", resized_frame)
        
        frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_cnt != -1 and frame_cnt == frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        cv2.imshow("Original Video", frame)
        sliced_frame = gray_slice(frame, min_slice, max_slice)
        cv2.imshow("Sliced Video", sliced_frame)
        
        test_contrast[:,:] = back_gray
        test_contrast[200:600,100:700] = fore_gray        
        
        for i in range(gradient.shape[1]):
            fraction = i / (gradient.shape[1]-1)
            float_gradient[:,i:(i+1)] = fraction*max_gray
            
        gradient = cv2.convertScaleAbs(float_gradient)
                    
        #cv2.imshow("Gradient", gradient)
        #cv2.imshow("Float Gradient", float_gradient/255.0)
        #cv2.imshow("Contrast", test_contrast)
        
        key = cv2.waitKey(33)
        
        if key == ord('y'): scale += 1
        if key == ord('h'): scale = max(1, scale-1)
        print(scale)
        
        if key == ord('r'): min_slice += 5
        if key == ord('f'): min_slice -= 5
        
        if key == ord('t'): max_slice += 5
        if key == ord('g'): max_slice -= 5
        #print(min_slice, max_slice)
        
        if key == ord('w'): fore_gray = np.clip(fore_gray+1, 0, 255)
        if key == ord('s'): fore_gray = np.clip(fore_gray-1, 0, 255)
                
        if key == ord('e'): back_gray = np.clip(back_gray+1, 0, 255)
        if key == ord('d'): back_gray = np.clip(back_gray-1, 0, 255)
        
        #print(fore_gray, back_gray)
        
        if key == ord('q'): max_gray += 5
        if key == ord('a'): max_gray -= 5
        #print(max_gray)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    