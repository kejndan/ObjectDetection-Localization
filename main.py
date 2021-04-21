import torch
from config import cfg
from Models.conv_net_v5 import ConvNetV5
from Train.train import train
from Losses.LocalizationLoss import LocalizationLoss
from torchsummary import summary
from Models.MobileNet_exp_1 import ConvNet_w_MN_small_v3
if __name__ == '__main__':
    print()
    loss_localization = LocalizationLoss().to(cfg.device)
    loss_classification = torch.nn.BCEWithLogitsLoss().to(cfg.device)
    model = ConvNet_w_MN_small_v3().to(cfg.device)

    print(summary(model,(3,220,220)))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train(model, [loss_localization, loss_classification], optimizer, cfg)