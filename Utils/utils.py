import numpy as np
import torch
import os

def IoU(prediction_box, ground_truth_box, cfg):
    """
    This function calculates mean Intersection over Union for numpy arrays and torch tensors
    :param prediction_box: coords of prediction boxes [x_left_up, y_left_up, x_right_down, y_right_down]
    :param ground_truth_box: coords of ground truth boxes [x_left_up, y_left_up, x_right_down, y_right_down]
    :return: mean IoU
    """
    if isinstance(prediction_box, torch.Tensor):  # if the torch tensor was passed
        prediction_box = prediction_box.type(torch.int32)
        ground_truth_box = ground_truth_box.type(torch.int32)

        # calculating the intersection area
        x_left_up = torch.max(prediction_box[:, 0], ground_truth_box[:, 0])
        y_left_up = torch.max(prediction_box[:, 1], ground_truth_box[:, 1])
        x_right_down = torch.min(prediction_box[:, 2], ground_truth_box[:, 2])
        y_right_down = torch.min(prediction_box[:, 3], ground_truth_box[:, 3])
        zero = torch.Tensor([0]).to(cfg.device)
        intersection = torch.max(zero, x_right_down - x_left_up + 1) * \
                       torch.max(zero, y_right_down - y_left_up + 1)
    elif isinstance(prediction_box, np.ndarray):  # if the numpy array was passed
        prediction_box = prediction_box.astype(np.int32)
        ground_truth_box = ground_truth_box.astype(np.int32)

        # calculating the intersection area
        x_left_up = np.maximum(prediction_box[:, 0], ground_truth_box[:, 0])
        y_left_up = np.maximum(prediction_box[:, 1], ground_truth_box[:, 1])
        x_right_down = np.minimum(prediction_box[:, 2], ground_truth_box[:, 2])
        y_right_down = np.minimum(prediction_box[:, 3], ground_truth_box[:, 3])
        intersection = np.maximum(0, x_right_down - x_left_up + 1) * np.maximum(0, y_right_down - y_left_up + 1)

    # calculating the union area
    square_p_box = (prediction_box[:, 2] - prediction_box[:, 0] + 1) * (prediction_box[:, 3] - prediction_box[:, 1] + 1)
    square_gt_box = (ground_truth_box[:, 2] - ground_truth_box[:, 0] + 1) \
                    * (ground_truth_box[:, 3] - ground_truth_box[:, 1] + 1)
    union = (square_p_box + square_gt_box - intersection)

    mean_iou = (intersection / union).mean()
    return mean_iou


def log_metric(name, iter, value, cfg):
    with open(os.path.join(cfg.path_to_logs, name), 'a') as f:
        f.write(f'{iter} {value} \n')

def calc_precision_recall(predict, ground_truth):
    correct_predict = predict == ground_truth

    true_positive = torch.logical_and(correct_predict == True, ground_truth == 1)
    false_positive = torch.logical_and(correct_predict == False, ground_truth == 0)
    false_negative = torch.logical_and(correct_predict == False, ground_truth == 1)

    temp1 = (true_positive + false_positive).sum().item()
    temp2 = (true_positive + false_negative).sum().item()

    precision = true_positive.sum().item()/temp1 if temp1 != 0 else 1
    recall = true_positive.sum().item()/temp2 if temp2 != 0 else 1


    return precision, recall