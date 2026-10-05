import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np
import os
from torchvision.io import decode_image

class CatDogDataset(Dataset):
    def __init__(self, root, train,
                 transform=None,
                 target_transform=None):
        
        if train:
            self.baseDir = os.path.join(root, "train")
        else:
            self.baseDir = os.path.join(root, "test")
        
        self.filenames = np.array(os.listdir(self.baseDir))
        self.transform = transform
        self.target_transform = target_transform
        self.classes = np.array(["cat", "dog"])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        if "cat" in filename:
            label = 0
        else:
            label = 1
        fullpath = os.path.join(self.baseDir, filename)
        image = decode_image(fullpath)
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return image, label

def main():
    data_transform = v2.Compose([
        v2.Resize((256,256)),
        # v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    data_root = os.path.join("data", "cats_dogs_light")
    training_data = CatDogDataset(root=data_root, 
                                     train=True, 
                                     transform=data_transform)
    
    testing_data = CatDogDataset(root=data_root, 
                                    train=False, 
                                    transform=data_transform)
    
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
            #img = cv2.resize(img, dsize=(256,256))
                        
            label_index = y[i]
            label = training_data.classes[label_index]
            window_name = "Img%02d_%s" % (i, label)
            cv2.imshow(window_name, img)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
            
        
        
    
    

if __name__ == "__main__":
    main()
    