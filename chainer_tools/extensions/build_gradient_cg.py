from chainer.computational_graph import build_computational_graph
import chainer
import chainer.functions as F
import numpy as np
from chainer.training import extension
from chainer import function_hooks
from chainer import function

# install Chainer yuyu2172/computational-graph-edge-attributes


class GradientLogHook(function.FunctionHook):

    def __init__(self, grad_dict,  precision=4):
        self.grad_dict = grad_dict
        self.precision = precision

    def backward_postprocess(self, function, in_data, out_grad):
        gxs = function.backward(in_data, out_grad)
        gnorm_means = [np.mean(np.abs(gx)) for gx in gxs if gx is not None]
        gnorm_mean_strs =\
            ['{:.{precision}f}'.format(gnorm_mean, precision=self.precision)
             for gnorm_mean in gnorm_means]
        keys = [(x, function) for x in function.inputs]
        attributes = [{'label': gnorm_mean_str, 'mean': gnorm_mean_str}
                      for gnorm_mean_str in gnorm_mean_strs]
        self.grad_dict.update(
            dict(zip(keys, attributes)))


class BuildGradientCG(extension.Extension):

    def __init__(self, out_file='cg_{.updater.iteration}', precision=4):
        self.out_file = out_file
        self.precision = precision

    def __call__(self, trainer):
        updater = trainer.updater
        model = updater._optimizers['main'].target
        in_arrays = updater.converter(updater._iterators['main'].next(),
                                    updater.device)
        #print in_arrays.shape
        grad_dict = {}
        with GradientLogHook(grad_dict, self.precision):
            loss = model(*in_arrays)
            loss.backward()
        cg = build_computational_graph(outputs=[loss], edge_attributes=grad_dict)
        with open(self.out_file.format(trainer), 'w') as o:
            o.write(cg.dump())

