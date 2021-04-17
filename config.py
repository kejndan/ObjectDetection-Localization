from easydict import EasyDict
import os

cfg = EasyDict()
cfg.random_seed = 0
cfg.device = 'cuda:0'

cfg.save_logs = True

cfg.batch_size = 8
cfg.lr = 1e-3
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.total_epochs = 100

cfg.threshold_classifier = 0.5

cfg.name_experiment = 'classif_1_exp'
cfg.path_to_dataset = 'C:\\Users\\adels\PycharmProjects\datasets\cats_dogs_220_balanced'
cfg.path_to_logs = os.path.join('C:\\Users\\adels\PycharmProjects\ObjectDetection&Localization',
                                'logs', cfg.name_experiment)

if not os.path.exists(cfg.path_to_logs):
    os.makedirs(cfg.path_to_logs)
