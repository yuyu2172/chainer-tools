import copy

import chainer
from chainer.dataset import convert

import visdom


class VisdomReport(chainer.training.extension.Extension):

    def __init__(self, iterator, target, global_dict,
                 env_name_suffix=None):
        self.iterator = copy.copy(iterator)
        self.target = target
        self.env_name_suffix = env_name_suffix
        self.global_dict = global_dict

    def __call__(self, trainer):
        device = trainer.updater.device
        try:
            batch = self.iterator.next()
        except StopIteration:
            self.iterator.reset()
            batch = self.iterator.next()

        in_arrays = convert.concat_examples(batch, device)
        in_vars = tuple(chainer.Variable(x) for x in in_arrays)

        env_name = 'gpu={0},iter={1:09}'.format(
            device, trainer.updater.iteration)
        if self.env_name_suffix is not None:
            env_name += self.env_name_suffix
        vis = visdom.Visdom(env=env_name)
        # close all previously created states
        vis.close()
        self.global_dict['visdom'] = vis
        self.global_dict['visualize'] = True
        self.target(*in_vars, vis=vis)
        self.global_dict['visualize'] = False
