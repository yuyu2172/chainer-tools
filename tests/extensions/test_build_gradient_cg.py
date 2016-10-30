import unittest
from chainer_tools.extensions.build_gradient_cg import BuildGradientCG


import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


if __name__ == '__main__':
    batchsize = 1
    unit = 1000
    gpu = -1
    model = L.Classifier(MLP(unit, 10))
    #if args.gpu >= 0:
    #    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    #    model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)


    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, batchsize)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (2, 'iteration'), out='result')
    trainer.extend(BuildGradientCG(), trigger=(1, 'iteration'))
    trainer.run()
