import torch
from Utils.utils import IoU, log_metric, calc_precision_recall


def evaluate(data_loader, model,  epoch, name_data, cfg, criterion=None):
    print('Test')
    model.eval()
    accuracy_sum = 0
    iou_sum = 0
    precision_sum = 0
    recall_sum = 0
    total_iter = len(data_loader)
    count_iter = 0
    for iteration, batch in enumerate(data_loader):
        if cfg.overfitting_for_some_batches:
            if cfg.batches_for_learning < iteration + 1:
                break
        images = batch[0].to(cfg.device)
        targets = batch[1].to(cfg.device)
        count_iter += 1

        with torch.no_grad():
            output = model(images)
            if criterion is not None:
                loss_localization = criterion[0](output[:, 1:], targets[:, 1:])
                loss_classification = criterion[1](output[:, 0], targets[:, 0])
                loss = loss_localization + loss_classification

            convert_coord = output[:, 1:].clone()
            convert_coord[:, [0, 2]] *= images[0].size(1)
            convert_coord[:, [1, 3]] *= images[0].size(2)
            iou = IoU(convert_coord, targets[:, 1:], cfg).item()
            predict = torch.where(output[:, 0] >= cfg.threshold_classifier, 1, 0)

            accuracy = (predict == targets[:, 0]).type(torch.float16).mean().item()

            precision, recall = calc_precision_recall(predict, targets[:, 0])

        if cfg.save_logs:
            log_metric(f'{name_data}_loss_localization', epoch * total_iter + iteration, loss_localization.item(), cfg)
            log_metric(f'{name_data}_loss_classification', epoch * total_iter + iteration, loss_classification.item(), cfg)
            log_metric(f'{name_data}_loss', epoch * total_iter + iteration, loss.item(), cfg)

        precision_sum += precision
        recall_sum += recall
        iou_sum += iou
        accuracy_sum += accuracy

    print(f'mIoU = {iou_sum/ count_iter}; Accuracy = {accuracy_sum/ count_iter}; \n'
          f'Precision = {precision_sum/ count_iter}; Recall = {recall_sum/ count_iter}')

    if cfg.save_logs:
        log_metric(f'{name_data}_mIoU', epoch, iou_sum / total_iter, cfg)
        log_metric(f'{name_data}_accuracy', epoch, accuracy_sum / total_iter, cfg)
        log_metric(f'{name_data}_precision', epoch, precision_sum / total_iter, cfg)
        log_metric(f'{name_data}_recall', epoch, recall_sum / total_iter, cfg)

    return  iou_sum / total_iter, accuracy_sum / total_iter
