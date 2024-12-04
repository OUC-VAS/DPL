'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''

import torch
import random
import numpy as np
from .abstract_dataset import DeepfakeAbstractBaseDataset
import os

class pairDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real and fake image lists
        # Fix the label of real images to be 0 and fake images to be 1
        self.fake_imglist = [(img, label, 1) for img, label in zip(self.image_list, self.label_list) if label != 0]
        self.real_imglist = [(img, label, 0) for img, label in zip(self.image_list, self.label_list) if label == 0]

        #获得根路径
        self.dataset_root_path = config['dataset_path_prefix']

    def __getitem__(self, index):
        # Get the fake and real image paths and labels
        fake_image_path, fake_spe_label, fake_label = self.fake_imglist[index]
        real_index = random.randint(0, len(self.real_imglist) - 1)  # Randomly select a real image
        real_image_path, real_spe_label, real_label = self.real_imglist[real_index]

        fake_image_path = os.path.join(self.dataset_root_path, fake_image_path)
        real_image_path = os.path.join(self.dataset_root_path, real_image_path)

        # Load the fake and real images
        fake_image = self.base_transform(self.load_rgb(fake_image_path))
        real_image = self.base_transform(self.load_rgb(real_image_path))

        fake_image = np.array(fake_image)  # Convert to numpy array for data augmentation
        real_image = np.array(real_image)  # Convert to numpy array for data augmentation

        fake_image_trans_list = []
        real_image_trans_list = []

        for compress_level in range(0, self.compress_num): # do compression only in training stage
            # Do transforms for fake and real images
            fake_image_trans, fake_jpeg_quality = self.data_aug(fake_image, compress_level)
            real_image_trans, real_jpeg_quality = self.data_aug(real_image, compress_level)

            # To tensor for fake and real images
            fake_image_trans = self.to_tensor(fake_image_trans)
            real_image_trans = self.to_tensor(real_image_trans)

            fake_image_trans_list.append(fake_image_trans)
            real_image_trans_list.append(real_image_trans)


        return {"fake": (fake_image_trans_list, fake_label), 
                "real": (real_image_trans_list, real_label)}

    def __len__(self):
        return len(self.fake_imglist)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                        the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images_tuple, fake_labels = zip(*[data["fake"] for data in batch])
        real_images_tuple, real_labels = zip(*[data["real"] for data in batch])

        # Stack the image, label, landmark, and mask tensors for fake and real data
        fake_images = [torch.stack([img_list[i] for img_list in fake_images_tuple], dim=0) for i in range(len(fake_images_tuple[0]))]
        fake_labels = torch.LongTensor(fake_labels)
        real_images = [torch.stack([img_list[i] for img_list in real_images_tuple], dim=0) for i in range(len(real_images_tuple[0]))]
        real_labels = torch.LongTensor(real_labels)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = [torch.cat([real_img, fake_img], dim=0) for real_img, fake_img in zip(real_images, fake_images)]
        labels = torch.cat([real_labels, fake_labels], dim=0)

        data_dict = {
            'image': images,
            'label': labels,
        }
        return data_dict
