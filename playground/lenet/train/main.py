import mindspore.nn as nn
from mindspore import context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from playground.lenet.train import train_net, test_net, LeNet5

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target='CPU')
    dataset_sink_mode = False

    # download mnist dataset
    # download_dataset()

    # learning rate setting
    lr = 0.01
    momentum = 0.9
    epoch_size = 1
    mnist_path = "/playground/lenet/MNIST_Data"

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    repeat_size = epoch_size

    # create the network
    network = LeNet5()

    # define the optimizer
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

    # save the network model and parameters for subsequence fine-tuning
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

    # group layers into an object with training and evaluation features
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    train_net(model, epoch_size, mnist_path, repeat_size, ckpoint_cb, dataset_sink_mode)
    test_net(network, model, mnist_path)
