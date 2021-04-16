import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms


class CatsDogs(torch.utils.data.Dataset):
    def __init__(self, path, cfg, work_mode='train', transform_mode='train'):
        """
        This class is used to create a dataset.
        :param path: path to images and annotations
        :param cfg: config
        :param work_mode:
         @train - the part of the dataset on which the training will take place;
         @valid - the part of the dataset where the selection of hyperparameters and other settings
         @test - the part of the dataset where the final accuracy will be measured
         @full_test - this is a combination of the validation and testing parts, if you don't need separation
        :param transform_mode: for selection augmentation
        """

        self.path_dir = path
        self.work_mode = work_mode
        self.transform_mode = transform_mode
        self.cfg = cfg

        self.paths_to_imgs = []
        self.__get_paths_to('.jpg', self.paths_to_imgs)
        self.paths_to_imgs.sort()

        self.paths_to_target = []
        self.__get_paths_to('.txt', self.paths_to_target)
        self.paths_to_target.sort()

        self.nb_classes = 2
        self.nb_samples = 3385

        self.__split_and_take(work_mode)

        self.crop_size = 32
        self.angle_rotation = 30
        self.sz_resize = 220
        self.means = np.array((0.4914, 0.4822, 0.4465))
        self.stds = np.array((0.247, 0.243, 0.262))

    def __get_paths_to(self, ext, paths):
        """
        This function is used to extract paths
        :param ext:
        @.jpg - get paths to images
        @.txt - get paths to annotations
        :param paths: list for storing paths
        """
        for file in os.listdir(self.path_dir):
            if file.endswith(ext):
                paths.append(os.path.join(self.path_dir, file))

    def __split_and_take(self, name_part):
        """
        This function is used to separate and get the training/validation/test part from the dataset
        :param name_part: names of the part to get
        """
        train_x, full_test_x, train_y, full_test_y =\
            train_test_split(self.paths_to_imgs, self.paths_to_target, test_size=0.2, random_state=self.cfg.random_seed)

        train_x, full_test_x, train_y, full_test_y = train_test_split(self.paths_to_imgs, self.paths_to_target,
                                                                      test_size=0.2, random_state=self.cfg.random_seed)

        if name_part == 'train':
            self.imgs = train_x
            self.target = train_y
        elif name_part == 'valid' or name_part == 'test':
            valid_x, test_x, valid_y, test_y = train_test_split(full_test_x, full_test_y, test_size=0.5,
                                                                random_state=self.cfg.random_seed)
            self.imgs = valid_x if name_part == 'valid' else test_x
            self.target = valid_y if name_part == 'valid' else test_y
        elif name_part == 'full_test':
            self.imgs = full_test_x
            self.target = full_test_y

    def get_labels_classes(self):
        labels = []
        for path_to_file in self.target:
            with open(path_to_file,'r') as f:
                label = int(f.read().split()[0])
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        if len(list(img.split())) != 3:
            img = img.convert('RGB')
        img = self.apply_augmentation(img)
        with open(self.target[idx],'r') as f:
            target = np.array(f.read().split()).astype('int32')
        target[0] -= 1
        return img, torch.Tensor(target)

    def apply_augmentation(self, img):
        if self.transform_mode == 'train':
            transforms_ = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.means,
                    std=self.stds,
                )
            ])
        else:
            transforms_ = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.means,
                    std=self.stds,
                )
            ])
        return transforms_(img)





