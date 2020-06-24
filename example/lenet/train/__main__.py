import mindspore.nn as nn
from mindinsight.lineagemgr import TrainLineage
from mindspore import context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, SummaryStep
from mindspore.train.summary import SummaryRecord

from example.lenet.train import download_dataset
from playground.lenet.train import train_net, test_net, LeNet5


def train_lenet():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target="CPU")
    dataset_sink_mode = False

    # download mnist dataset
    download_dataset()

    # learning rate setting
    lr = 0.01
    momentum = 0.9
    epoch_size = 1
    mnist_path = "../MNIST_Data"

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
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

    summary_writer = SummaryRecord(log_dir="../../summary", network=network)
    summary_callback = SummaryStep(summary_writer, flush_step=10)

    # Init TrainLineage to record the training information
    train_callback = TrainLineage(summary_writer)

    train_net(
        model,
        epoch_size,
        mnist_path,
        repeat_size,
        ckpoint_cb,
        dataset_sink_mode,
        callbacks=[summary_callback, train_callback],
    )

    test_net(network, model, mnist_path)

    summary_writer.close()
