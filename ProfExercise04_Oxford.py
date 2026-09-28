import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np

def main():
    data_transform = v2.Compose([
        v2.Resize((256,256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    target_transform = v2.Compose([
        v2.Resize((256,256),
                  interpolation=v2.InterpolationMode.NEAREST),
        v2.ToImage()
    ])
    
    training_data = datasets.OxfordIIITPet(root="data", 
                                     split="trainval", 
                                     transform=data_transform,
                                     target_transform=target_transform,
                                     target_types="segmentation",
                                     download=True)
    
    testing_data = datasets.OxfordIIITPet(root="data", 
                                     split="test", 
                                     transform=data_transform,
                                     target_transform=target_transform,
                                     target_types="segmentation",
                                     download=True)
    
    batch_size = 5
    train_ds = DataLoader(training_data, 
                          batch_size=batch_size,
                          shuffle=True)
    
    test_ds = DataLoader(testing_data, 
                            batch_size=batch_size,
                            shuffle=False)
    
    train_iter = iter(train_ds)
    
    segLUT = np.array([
        [0,0,0],
        [255,0,0],
        [0,255,0],
        [0,0,255]
    ], dtype="uint8")
    
    for _ in range(3):
        X,y = next(train_iter)
        X = X.numpy()
        y = y.numpy()
        for i in range(batch_size):
            img = X[i]
            img = np.transpose(img, [1,2,0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            seg = y[i]
            seg = np.transpose(seg, [1,2,0])
            seg = np.squeeze(seg, axis=-1)
            seg = segLUT[seg]
                 
            image_name = "Img%02d" % i
            seg_name = "Seg%02d" % i
            cv2.imshow(image_name, img)
            cv2.imshow(seg_name, seg)
            
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
            
        
        
    
    

if __name__ == "__main__":
    main()
    