import numpy
from Dataset.cats_n_dogs import CatsDogs
from config import cfg
from network import NeuralNetwork
from Layers import Conv2d
from Layers import Dropout
from Layers import Flatten
from Layers import Linear
from Layers import Pool
from Layers import activation
from Layers import losses
import numpy as np


def get_batch(iteration, dataset, batch_size=8):
    x_batch = []
    y_batch = []
    for i in range(iteration*batch_size,(iteration+1)*batch_size):
        sample = dataset[i]
        x_batch.append(sample[0])
        y_batch.append(sample[1])
    return np.array(x_batch), np.array(y_batch)

if __name__ == '__main__':
    dataset = CatsDogs(cfg.path_to_dataset,cfg,framework='numpy')
    model = NeuralNetwork()
    model.add_module(Conv2d.Conv2d(3, 8, 41, 2))
    model.add_module(activation.ReLU())
    model.add_module(Conv2d.Conv2d(8,8,19,2))
    model.add_module(activation.ReLU())
    model.add_module(Pool.Pool(2,stride=2))
    model.add_module(Dropout.Dropout(0.2))

    model.add_module(Conv2d.Conv2d(8, 16, 3))
    model.add_module(activation.ReLU())
    model.add_module(Conv2d.Conv2d(16,16,3))
    model.add_module(activation.ReLU())
    model.add_module(Pool.Pool(2,stride=2))
    model.add_module(Dropout.Dropout(0.2))

    model.add_module(Flatten.Flatten())
    model.add_module(Linear.Linear(784, 512))
    model.add_module(Dropout.Dropout(0.5))
    model.add_module(Linear.Linear(512,5))

    model.add_loss_function([losses.BCEWithLogitsLoss(),losses.LocalizationLoss()])

    batch_size = 8
    epoch = 100

    for e in range(epoch):
        for iter in range(len(dataset)//batch_size):

            batch = get_batch(iter,dataset,batch_size)
            print(batch[0].shape)
            target = model.forward(batch[0])
            print(target)



