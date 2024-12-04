import sys
sys.path.append('.')

import os
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

# refer to DeepfakeBench: https://github.com/SCLBD/DeepfakeBench
class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train', jpeg_compress_factor=100):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """
        
        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        self.jpeg_compress_factor = jpeg_compress_factor
        # Dataset dictionary
        self.image_list = []
        self.label_list = []

        self.compress_num = config['compress_num']

        if self.compress_num > 1:
            self.compress_range_list = [(30, 99), (13, 29), (8, 12)]

        self.dataset_root_path = config['dataset_path_prefix']
        
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
        elif mode == 'test':
            dataset_list = config['test_dataset']
        elif mode == 'val':
            dataset_list = config['valid_dataset']
        else:
            raise NotImplementedError('Only train and test modes are supported.')
        self.dataset_list = dataset_list
        
        # Collect image and label lists
        image_list, label_list = self.collect_img_and_label(dataset_list)
        self.image_list, self.label_list = image_list, label_list
                    
        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }

        self.transform_list = self.create_transform_list()
        self.base_transform = T.CenterCrop(size=224)
    
    def create_transform_list(self):
        trans = []
        for i in range(1, self.compress_num):
            item = self.compress_range_list[i-1]
            trans.append(A.augmentations.transforms.ImageCompression(quality_lower=item[0], quality_upper=item[1], p=1))
        return trans
       
    def collect_img_and_label(self, dataset_list):
        """Collects image and label lists.

        Args:
            dataset_list (dict): A dictionary containing dataset information.

        Returns:
            list: A list of image paths.
            list: A list of labels.
        
        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []

        # If the dataset dictionary is not empty, collect the image, label, landmark, and mask lists
        if dataset_list:
            # Iterate over the datasets in the dictionary
            for dataset_name in dataset_list:
                cp = None
                if dataset_name == 'FaceForensics++_c40':
                    dataset_name = 'FaceForensics++'
                    cp = 'c40'
                elif dataset_name == 'FF-DF_c40':
                    dataset_name = 'FF-DF'
                    cp = 'c40'
                elif dataset_name == 'FF-F2F_c40':
                    dataset_name = 'FF-F2F'
                    cp = 'c40'
                elif dataset_name == 'FF-FS_c40':
                    dataset_name = 'FF-FS'
                    cp = 'c40'
                elif dataset_name == 'FF-NT_c40':
                    dataset_name = 'FF-NT'
                    cp = 'c40'
                # Try to get the dataset information from the JSON file
                try:
                    with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                        dataset_info = json.load(f)
                except:
                    print(f'dataset {dataset_name} not exist! Lets skip it.')
                    continue # skip if the dataset does not exist in the default image path

                # If JSON file exists, the following processes the data according to your original code
                else:
                    # Get the information for the current dataset
                    for label in dataset_info[dataset_name]:
                        sub_dataset_info = dataset_info[dataset_name][label][self.mode]
                        # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
                        if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                            sub_dataset_info = sub_dataset_info[self.compression]
                        elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                            sub_dataset_info = sub_dataset_info['c40']
                        # Iterate over the videos in the dataset
                        for video_name, video_info in sub_dataset_info.items():
                            # Get the label and frame paths for the current video
                            if video_info['label'] not in self.config['label_dict']:
                                raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                            label = self.config['label_dict'][video_info['label']]
                            frame_paths = video_info['frames']

                            # Select self.frame_num frames evenly distributed throughout the video
                            total_frames = len(frame_paths)
                            if self.frame_num < total_frames:
                                # if self.mode == 'train' and self.compression == 'c23':
                                if self.mode == 'train' and cp is None:
                                    sorted_frame_paths = sorted(frame_paths, key=lambda x: x['path'])
                                else:
                                    sorted_frame_paths = sorted(frame_paths)
                                step = total_frames // self.frame_num
                                selected_frames = [sorted_frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]
                            else:
                                selected_frames = deepcopy(frame_paths)
                                
                            
                            if self.mode == 'train' and cp is None: 
                                # Append the label and frame paths to the lists
                                label_list.extend([label]*len(selected_frames))
                                for item in selected_frames:
                                    frame_path_list.append(item['path'])
                            else:
                                label_list.extend([label]*len(selected_frames))
                                frame_path_list.extend(selected_frames)

            if self.mode == 'train':
                # Shuffle the label and frame path lists in the same order
                shuffled = list(zip(label_list, frame_path_list))
                random.shuffle(shuffled)
                label_list, frame_path_list = zip(*shuffled)
            else:
                # Shuffle the label and frame path lists in the same order
                shuffled = list(zip(label_list, frame_path_list))
                random.shuffle(shuffled)
                label_list, frame_path_list = zip(*shuffled)
            
            return frame_path_list, label_list

        else:
            raise ValueError('No dataset is given.')
     
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None: 
            raise ValueError('Loaded image is None: {}'.format(file_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if os.path.exists(file_path):
            mask = cv2.imread(file_path, 0)
            if mask is None:
                mask = np.zeros((size, size))
            mask = cv2.resize(mask, (size, size))/255
            mask = np.expand_dims(mask, axis=2)
            return np.float32(mask)
        else:
            return np.zeros((size, size, 1))

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if os.path.exists(file_path):
            landmark = np.load(file_path)
            return np.float32(landmark)
        else:
            return np.zeros((81, 2))

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def image_compression(self, img, quality, image_type):
        """refer to albumentations (https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py#L475)"""
        if quality == 100:
            return img
        
        if image_type in [".jpeg", ".jpg"]:
            quality_flag = cv2.IMWRITE_JPEG_QUALITY
        elif image_type == ".webp":
            quality_flag = cv2.IMWRITE_WEBP_QUALITY
        else:
            NotImplementedError("Only '.jpg' and '.webp' compression transforms are implemented. ")

        _, encoded_img = cv2.imencode(image_type, img, (int(quality_flag), quality))
        img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

        return img

    def data_aug(self, img, compress_level=0):
        """
        Apply data augmentation to an image, landmark, and mask.
        """
        
        # Create a dictionary of arguments
        kwargs = {'image': img}

        jpeg_quality = 100
        
        if compress_level > 0:
            transform = self.transform_list[compress_level-1]
            compress_params = transform.get_params()
            jpeg_quality = compress_params['quality']
            augmented_img = self.image_compression(kwargs['image'], compress_params['quality'], compress_params['image_type'])
        else:
            augmented_img = deepcopy(img)

        return augmented_img, jpeg_quality

    def __getitem__(self, index):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_path = self.data_dict['image'][index]
        label = self.data_dict['label'][index]
        image_path = os.path.join(self.dataset_root_path, image_path)
        
        # Load the image
        try:
            image = self.base_transform(self.load_rgb(image_path))
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)  # Convert to numpy array for data augmentation
        image_trans_list = []

        # Do transforms 
        image_trans = deepcopy(image)
        quality_flag = '.jpg'
        image_trans = self.image_compression(image_trans, self.jpeg_compress_factor, quality_flag)
        # To tensor
        image_trans = self.to_tensor(image_trans)
        image_trans_list.append(image_trans)
        
        return image_trans_list, label
    
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
        # Separate the image, label, landmark, and mask tensors
        images_tuple, labels = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = [torch.stack([img_list[i] for img_list in images_tuple], dim=0) for i in range(len(images_tuple[0]))]
        labels = torch.LongTensor(labels)
        
        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images[0]
        data_dict['label'] = labels
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)