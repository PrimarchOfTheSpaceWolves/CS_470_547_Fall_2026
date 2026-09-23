import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np

def main():
    data_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    training_data = datasets.CIFAR10(root="data", 
                                     train=True, 
                                     transform=data_transform,
                                     download=True)
    
    testing_data = datasets.CIFAR10(root="data", 
                                    train=False, 
                                    transform=data_transform,
                                    download=True)
    
    batch_size = 5
    train_ds = DataLoader(training_data, 
                          batch_size=batch_size,
                          shuffle=True)
    
    test_ds = DataLoader(testing_data, 
                            batch_size=batch_size,
                            shuffle=False)
    
    train_iter = iter(train_ds)
    
    for _ in range(3):
        X,y = next(train_iter)
        X = X.numpy()
        y = y.numpy()
        for i in range(batch_size):
            img = X[i]
            img = np.transpose(img, [1,2,0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, dsize=(256,256))
                        
            label_index = y[i]
            label = training_data.classes[label_index]
            window_name = "Img%02d_%s" % (i, label)
            cv2.imshow(window_name, img)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
            
        
        
    
    

if __name__ == "__main__":
    main()
    