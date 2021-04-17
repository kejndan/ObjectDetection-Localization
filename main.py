import torch
from config import cfg
from Models.conv_net_v1 import ConvNetV1
from Train.train import train
from Losses.LocalizationLoss import LocalizationLoss

if __name__ == '__main__':
    loss_localization = LocalizationLoss().to(cfg.device)
    loss_classification = torch.nn.BCEWithLogitsLoss().to(cfg.device)
    model = ConvNetV1().to(cfg.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    train(model, [loss_localization, loss_classification], optimizer, cfg)