import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np

def main():
    base_transform = v2.Compose([
        v2.Resize((256,256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    aug_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop((256,256)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    def all_base_transform(image, targets):
        seg, label = targets
        seg = torchvision.tv_tensors.Mask(seg)
        image, seg = base_transform(image, seg)
        return image,(seg, label)
    
    def all_aug_transform(image, targets):
        seg, label = targets
        seg = torchvision.tv_tensors.Mask(seg)
        image, seg = aug_transform(image, seg)
        return image,(seg, label)
        
    training_data = datasets.OxfordIIITPet(root="data", 
                                     split="trainval", 
                                     transforms=all_aug_transform,
                                     target_types=["segmentation","category"],
                                     download=True)
    
    testing_data = datasets.OxfordIIITPet(root="data", 
                                     split="test", 
                                     transforms=all_base_transform,
                                     target_types=["segmentation","category"],
                                     download=True)
    training_data = datasets.wrap_dataset_for_transforms_v2(training_data)
    testing_data = datasets.wrap_dataset_for_transforms_v2(testing_data)
            
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
        X,(y,labels) = next(train_iter)
        X = X.numpy()
        y = y.numpy()
        labels = labels.numpy()
        for i in range(batch_size):
            img = X[i]
            img = np.transpose(img, [1,2,0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            seg = y[i]
            seg = np.transpose(seg, [1,2,0])
            seg = np.squeeze(seg, axis=-1)
            seg = segLUT[seg]
            
            label_name = training_data.classes[labels[i]]
                 
            image_name = "Img%02d_%s" % (i, label_name)
            seg_name = "Seg%02d_%s" % (i, label_name)
            cv2.imshow(image_name, img)
            cv2.imshow(seg_name, seg)
            
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
            
        
        
    
    

if __name__ == "__main__":
    main()
    