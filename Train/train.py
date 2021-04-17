import torch
from Dataset.cats_n_dogs import CatsDogs
from torch.utils.data import DataLoader
from Train.evaluate import evaluate
from Utils.utils import IoU, log_metric


def train(model, criterion, optimizer, cfg, manual_load=None):
    ds_train = CatsDogs(cfg.path_to_dataset, cfg, work_mode='train', transform_mode='train')
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size)

    ds_valid = CatsDogs(cfg.path_to_dataset, cfg, work_mode='valid', transform_mode='test')
    dl_valid = DataLoader(ds_valid, batch_size=32, shuffle=True)

    for epoch in range(cfg.total_epochs):
        training_epoch(dl_train, model, criterion, optimizer, epoch, cfg)
        evaluate(dl_train, model, criterion, epoch, 'train', cfg)
        evaluate(dl_valid, model, criterion, epoch, 'validation', cfg)


def training_epoch(data_loader, model, criterion, optimizer, epoch, cfg):
    print('Train')
    model.train()
    accuracy_sum = 0
    iou_sum = 0
    loss_sum = 0
    total_iter = len(data_loader)


    for iteration, batch in enumerate(data_loader):


        images = batch[0].to(cfg.device)
        targets = batch[1].to(cfg.device)

        output = model(images)

        loss_localization = criterion[0](output[:, 1:], targets[:, 1:])
        loss_classification = criterion[1](output[:, 0], targets[:, 0])
        loss = loss_classification
        print(loss.item())

        convert_coord = output[:, 1:].clone()
        convert_coord[:, [0, 2]] *= criterion[0].size_img[0]
        convert_coord[:, [1, 3]] *= criterion[0].size_img[1]
        iou = IoU(convert_coord, targets[:, 1:], cfg)
        accuracy = (torch.sign(output[:, 0]) == targets[:, 0]).type(torch.float16).mean()
        iou_sum += iou
        accuracy_sum += accuracy
        loss_sum += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % 50 == 0:
            print(
                f'Epoch: {epoch}. Iteration {iteration} of {total_iter}. {loss_sum.item() / 50, iou_sum.item() / 50, accuracy_sum.item() / 50}')
            accuracy_sum = 0
            iou_sum = 0
            loss_sum = 0

        if cfg.save_logs:
            log_metric('train_loss_localization', epoch * total_iter + iteration, loss_localization.item(), cfg)
            log_metric('train_loss_classification', epoch * total_iter + iteration, loss_classification.item(), cfg)
            log_metric('train_loss', epoch * total_iter + iteration, loss.item(), cfg)



if __name__ == '__main__':
    print('Train module')