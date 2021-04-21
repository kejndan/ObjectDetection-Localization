import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms


class CatsDogs(torch.utils.data.Dataset):
    def __init__(self, path, cfg, split_by = 'file', work_mode='train'):
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
        self.path_to_dataset = path
        self.path_dir = path + '\imgs'
        self.work_mode = work_mode
        self.cfg = cfg

        self.paths_to_imgs = []
        self.__get_paths_to('.jpg', self.paths_to_imgs)
        self.paths_to_imgs.sort()

        self.paths_to_target = []
        self.__get_paths_to('.txt', self.paths_to_target)
        self.paths_to_target.sort()

        self.nb_classes = 2
        self.nb_samples = 3385
        if split_by == 'file':
            self.__split_by_file(work_mode)
        else:
            self.__split_and_take(work_mode)

        self.crop_size = 32
        self.angle_rotation = 30
        self.sz_resize = 220
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

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

    def __split_by_file(self, name_part):
        imgs = []
        targets = []
        with open(os.path.join(self.path_to_dataset, f'name_imgs_{name_part}.txt')) as f:
            names_imgs = f.readlines()
        for name in names_imgs:
            imgs.append(name[:-1])
        with open(os.path.join(self.path_to_dataset, f'name_txts_{name_part}.txt')) as f:
            names_targets = f.readlines()
        for target in names_targets :
            targets.append(target[:-1])

        self.imgs = imgs
        self.target = targets


    def split_and_take_from_file(self):
        for name_part in ['train', 'valid','test']:

            with open(os.path.join(self.path_to_dataset, f'{name_part}_name.txt'),'r') as f:
                names = f.readlines()
            for name in names:
                name = name[:-1]
                for name_ in self.paths_to_imgs:
                    if name+'.jpg' in name_ or name+'_aug' in name_:
                        with open(os.path.join(self.path_to_dataset,f'name_imgs_{name_part}.txt'),'a') as f:
                            f.write(name_+'\n')
                        with open(os.path.join(self.path_to_dataset,f'name_txts_{name_part}.txt'),'a') as f:
                            f.write(name_[:-4]+'.txt'+'\n')

    def save_name_files(self, name_part, path):
        train_x, full_test_x, train_y, full_test_y = train_test_split(self.paths_to_imgs, self.paths_to_target,
                                                                      test_size=0.2, random_state=self.cfg.random_seed)

        if name_part == 'train':
            names = train_x
        elif name_part == 'valid' or name_part == 'test':
            valid_x, test_x, valid_y, test_y = train_test_split(full_test_x, full_test_y, test_size=0.5,
                                                                random_state=self.cfg.random_seed)
            names = valid_x if name_part == 'valid' else test_x
        elif name_part == 'full_test':
            names = full_test_x

        with open(os.path.join(path, f'{name_part}_name.txt'), 'a') as f:
            for name in names:
                f.write(name[name.rfind('\\')+2:-4]+'\n')



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
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.means,
                std=self.stds,
            )
        ])
        return transforms_(img)









