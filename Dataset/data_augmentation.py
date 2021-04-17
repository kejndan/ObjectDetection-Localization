import os
from dicttoxml import dicttoxml
import json
from config import cfg
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import matplotlib.pyplot as plt
from random import randint


def resize_all_images(path, size=(220, 220)):
    """
    This function is needed to bring the images to the same size
    :param path: path of output
    :param size: size image
    """
    images = [file for file in os.listdir(cfg.path_to_dataset) if file.endswith('jpg')]
    for image in images:
        path_to_file = os.path.join(cfg.path_to_dataset,image)[:-4]
        img = np.array(Image.open(path_to_file+'.jpg'))

        with open(path_to_file+'.txt', 'r') as f:
            class_number, xmin, ymin, xmax, ymax = map(int, f.read().split())

        bb = BoundingBoxesOnImage([BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)], shape=img.shape)
        seq = iaa.Sequential([iaa.Resize({"height": size[0], "width": size[1]})])

        img_aug, bb_aug = seq(image=img, bounding_boxes=bb)
        bb_aug = bb_aug.bounding_boxes[0]

        coord_bb = [class_number, bb_aug.x1, bb_aug.y1, bb_aug.x2, bb_aug.y2]
        coord_bb = [int(s) for s in coord_bb]
        str_coord_bb = f'{coord_bb[0]} {coord_bb[1]} {coord_bb[2]} {coord_bb[3]} {coord_bb[4]}'

        im = Image.fromarray(img_aug)
        im.save(os.path.join(path, image))

        with open(os.path.join(path, image[:-4]+'.txt'),'w') as f:
            f.write(str_coord_bb)


def extending_dataset(path_in,path_out, size=(220, 220)):
    """
    This function is necessary for balancing the dataset,
     i.e., so that the number of images of different classes is approximately the same.
    :param path: path of output
    :param size: size image
    :return:
    """
    images = [file for file in os.listdir(path_in) if file.endswith('jpg')]
    for image in images:
        path_to_file = os.path.join(path_in,image)[:-4]

        with open(path_to_file+'.txt', 'r') as f:
            class_number, xmin, ymin, xmax, ymax = map(int, f.read().split())
        img = np.array(Image.open(path_to_file + '.jpg'))
        if class_number == 1:

            number_of_aug = randint(0, 3)

            bb = BoundingBoxesOnImage([BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)], shape=img.shape)

            if number_of_aug == 0:
                trans = iaa.Rot90(1)
            elif number_of_aug == 1:
                trans = iaa.Rot90(2)
            elif number_of_aug == 2:
                trans = iaa.Fliplr(1)
            elif number_of_aug == 3:
                trans = iaa.Flipud(1)
            seq = iaa.Sequential([trans])

            img_aug, bb_aug = seq(image=img, bounding_boxes=bb)
            bb_aug = bb_aug.bounding_boxes[0]
            coord_bb = [class_number, bb_aug.x1, bb_aug.y1, bb_aug.x2, bb_aug.y2]
            coord_bb = [int(s) if int(s) < size[0] else size[0] - 1 for s in coord_bb]
            str_coord_aug_bb = f'{coord_bb[0]} {coord_bb[1]} {coord_bb[2]} {coord_bb[3]} {coord_bb[4]}'


            im = Image.fromarray(img_aug)
            im.save(os.path.join(path_out,  image[:-4] + f'_{number_of_aug}' + image[-4:]))

            with open(os.path.join(path_out, image[:-4] + f'_{number_of_aug}' + '.txt'), 'w') as f:
                f.write(str_coord_aug_bb)

        im = Image.fromarray(img)
        im.save(os.path.join(path_out, image))
        str_coord_bb = f'{class_number} {xmin} {ymin} {xmax} {ymax}'
        with open(os.path.join(path_out, image[:-4] + '.txt'), 'w') as f:
            f.write(str_coord_bb)


def show_image_with_boxes(img, boxes):
    """
    This function is needed for drawing images with their bounding boxes.
    :param img: image
    :param boxes: bounding boxes [[xmin_1, ymin_1, xmax_1, ymax_1],...,[xmin_n, ymin_n, xmax_n, ymax_n]]
    """
    for i in range(len(boxes)):
        box = boxes[i]
        img[box[1]:box[3], box[0]] = [255, 255, 0]
        img[box[1], box[0]:box[2]] = [255, 255, 0]
        img[box[1]:box[3], box[2] - 1] = [255, 255, 0]
        img[box[3] - 1, box[0]:box[2]] = [255, 255, 0]
        plt.text(box[0], box[1], f'{np.round(boxes[i][1],2)} ({np.round(boxes[i][2],2)})', fontsize=5, color='black',
                 bbox={'facecolor':'yellow','edgecolor': 'yellow', 'boxstyle': 'round'})
    plt.title(f'Nums of boxes:{len(boxes)}')
    plt.imshow(np.uint8(img))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # create_xml_annotation('C:\\Users\\adels\PycharmProjects\ObjectDetection&Localization\Dataset\\annotation_template')
    # print(type(ia.quokka(size=(256, 256))))
    # resize_all_images('C:\\Users\\adels\PycharmProjects\datasets\cats_dogs_220')
    extending_dataset('C:\\Users\\adels\PycharmProjects\datasets\cats_dogs_220','C:\\Users\\adels\PycharmProjects\datasets\cats_dogs_220_balanced')