import torch
from Dataset.cats_n_dogs import CatsDogs
from torch.utils.data import DataLoader
from Train.evaluate import evaluate
from Utils.utils import IoU, log_metric
from Utils.utils_torch import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def train(model, criterion, optimizer, cfg):
    ds_train = CatsDogs(cfg.path_to_dataset, cfg, work_mode='train')
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size)

    ds_valid = CatsDogs(cfg.path_to_dataset, cfg, work_mode='valid')
    dl_valid = DataLoader(ds_valid, batch_size=32, shuffle=True)


    best_score = 0
    start_epoch = 0
    if cfg.load_save:
        model, optimizer, start_epoch, best_score = load_model(os.path.join(cfg.path_to_load,cfg.name_save),
                                                                 model,
                                                                 cfg,
                                                                 optimizer)
        if cfg.change_lr:
            for g in optimizer.param_groups:
                g['lr'] = cfg.lr

    if cfg.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-8,
                                                                  patience=cfg.ROP_patience,
                                                              factor=cfg.ROP_factor)
    if cfg.freeze_model:
        for name, param in model.named_parameters():
            if cfg.freeze_linear == 'classification' and 'linear_classify' in name:
                param.requires_grad = False
            elif cfg.freeze_linear == 'localization' and 'linear_localize' in name:
                param.requires_grad = False
            elif not ('linear_classify' in name or 'linear_localize' in name):
                param.requires_grad = False
            else:
                print(name)

    for epoch in range(start_epoch, cfg.total_epochs):
        print('Sleep')
        time.sleep(5)
        print('Awake')
        training_epoch(dl_train, model, criterion, optimizer, epoch, cfg)
        mIoU, accuracy=evaluate(dl_train, model,  epoch, 'train', cfg,criterion)

        if not cfg.overfitting_for_some_batches:
            mIoU, accuracy = evaluate(dl_valid, model,  epoch, 'validation', cfg,criterion)

        if cfg.use_lr_scheduler:
            lr_scheduler.step(mIoU)
            for g in optimizer.param_groups:
                print(g['lr'])
        if cfg.save_models or cfg.save_best_model:
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'best_score': best_score,
                'opt': optimizer.state_dict()
            }
            if cfg.save_models:
                torch.save(state, os.path.join(cfg.path_to_saves, f'checkpoint_{epoch}'))

            if cfg.save_best_model and (cfg.metric_for_best_model == 'mIoU' and best_score < mIoU
                                        or cfg.metric_for_best_model == 'accuracy' and best_score < accuracy):
                torch.save(state, os.path.join(cfg.path_to_saves, f'best_checkpoint'))
                best_score = mIoU if cfg.metric_for_best_model == 'mIoU' else accuracy

            if os.path.exists(os.path.join(cfg.path_to_saves, f'checkpoint_{epoch - 3}')):
                os.remove(os.path.join(cfg.path_to_saves, f'checkpoint_{epoch - 3}'))


def show_image_with_boxes(img, boxes):
    """
    This function is needed for drawing images with their bounding boxes.
    :param img: image
    :param boxes: bounding boxes [[xmin_1, ymin_1, xmax_1, ymax_1],...,[xmin_n, ymin_n, xmax_n, ymax_n]]
    """
    img *= 255
    for i in range(len(boxes)):
        box = boxes[i].astype(np.int32)
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


def training_epoch(data_loader, model, criterion, optimizer, epoch, cfg):
    print('Train')
    model.train()
    accuracy_sum = 0
    iou_sum = 0
    loss_sum = 0
    total_iter = len(data_loader)
    count_iter = 0
    for iteration, batch in enumerate(data_loader):
        if cfg.overfitting_for_some_batches:
            if cfg.batches_for_learning < iteration + 1:
                break

        count_iter += 1




        images = batch[0].to(cfg.device)

        targets = batch[1].to(cfg.device)

        output = model(images)

        loss_localization = criterion[0](output[:, 1:], targets[:, 1:])
        loss_classification = criterion[1](output[:, 0], targets[:, 0])

        if cfg.optimize_by == 'classification':
            loss = loss_classification
        elif cfg.optimize_by == 'localization':
            loss = loss_localization
        else:
            loss = loss_localization + loss_classification
        convert_coord = output[:, 1:].clone()
        convert_coord[:, [0, 2]] *= criterion[0].size_img[0]
        convert_coord[:, [1, 3]] *= criterion[0].size_img[1]
        iou = IoU(convert_coord, targets[:, 1:], cfg).item()
        predict = torch.where(output[:, 0] >= cfg.threshold_classifier, 1, 0)

        accuracy = (predict == targets[:, 0]).type(torch.float16).mean().item()
        iou_sum += iou
        accuracy_sum += accuracy
        loss_sum += loss


        optimizer.zero_grad()
        loss.backward()


        optimizer.step()
        if iteration % 50 == 0:
            print(
                f'Epoch: {epoch}. Iteration {iteration} of {total_iter}. {loss_sum.item()/ count_iter, iou_sum/ count_iter, accuracy_sum / count_iter}')
            accuracy_sum = 0
            iou_sum = 0
            loss_sum = 0
            count_iter = 0

        if cfg.save_logs:
            log_metric('train_loss_localization', epoch * total_iter + iteration, loss_localization.item(), cfg)
            log_metric('train_loss_classification', epoch * total_iter + iteration, loss_classification.item(), cfg)
            log_metric('train_loss', epoch * total_iter + iteration, loss.item(), cfg)





if __name__ == '__main__':
    print('Train module')