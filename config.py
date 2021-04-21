from easydict import EasyDict
import os

cfg = EasyDict()
cfg.random_seed = 0
cfg.device = 'cuda:0'

cfg.save_logs = True

cfg.batch_size = 8
cfg.lr = 1e-4
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.total_epochs = 1000

cfg.threshold_classifier = 0.5

cfg.use_lr_scheduler = True
cfg.ROP_patience = 5
cfg.ROP_factor = 0.1


cfg.name_experiment = 'Final_union'

cfg.path_to_dataset = 'C:\\Users\\adels\PycharmProjects\datasets\cats_dogs_220_ext'
cfg.path_to_logs = os.path.join('C:\\Users\\adels\PycharmProjects\ObjectDetection&Localization',
                                'logs', cfg.name_experiment)



cfg.path_to_saves = os.path.join('C:\\Users\\adels\PycharmProjects\ObjectDetection&Localization',
                                'saves', cfg.name_experiment)

cfg.path_to_load = cfg.path_to_saves #os.path.join('C:\\Users\\adels\PycharmProjects\ObjectDetection&Localization','saves', 'union')

cfg.save_models = True
cfg.save_best_model = True

cfg.load_save = True
cfg.name_save = 'best_checkpoint'
cfg.change_lr = True

cfg.metric_for_best_model = 'mIoU'#['accuracy', 'mIoU']

cfg.freeze_model = False
cfg.freeze_linear = 'localization' #['classification', 'localization']

cfg.optimize_by = 'union' # ['classification', 'localization', 'union']

#DEBUG
cfg.overfitting_for_some_batches = False
cfg.batches_for_learning = 2

if not os.path.exists(cfg.path_to_logs):
    os.makedirs(cfg.path_to_logs)

if not os.path.exists(cfg.path_to_saves):
    os.makedirs(cfg.path_to_saves)
