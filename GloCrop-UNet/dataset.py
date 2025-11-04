import os
import cv2
import numpy as np
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):

        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            ### Wheat
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i), img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])

            ### busi
            # mask.append(cv2.imread(os.path.join(self.mask_dir, str(i), img_id + '_mask' + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])

            ### chasedb1
            # mask.append(cv2.imread(os.path.join(self.mask_dir, str(i), img_id + '_1stHO' + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])

            ### glas
            # gt = (cv2.imread(os.path.join(self.mask_dir, str(i), img_id + '_anno' + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
            # mask.append(np.where(gt > 0, 255, gt))

        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)

            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)

        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
