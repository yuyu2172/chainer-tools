import copy

import chainer
from chainer.dataset import convert

import visdom


class VisdomReport(chainer.training.extension.Extension):

    def __init__(self, iterator, target, global_dict=None,
                 device=-1,
                 env_name_suffix=None,
                 pass_vis=False):
        self.iterator = copy.copy(iterator)
        self.target = target
        self.env_name_suffix = env_name_suffix
        self.global_dict = global_dict
        self.device = device
        self.pass_vis = pass_vis

    def __call__(self, trainer):
        try:
            batch = self.iterator.next()
        except StopIteration:
            self.iterator.reset()
            batch = self.iterator.next()

        in_arrays = convert.concat_examples(batch, self.device)
        in_vars = tuple(chainer.Variable(x) for x in in_arrays)

        env_name = 'gpu={0},iter={1:09}'.format(
            self.device, trainer.updater.iteration)
        if self.env_name_suffix is not None:
            env_name += self.env_name_suffix
        vis = visdom.Visdom(env=env_name)
        # close all previously created states
        vis.close()

        if self.global_dict is not None:
            self.global_dict['visdom'] = vis
            self.global_dict['visualize'] = True

        if self.pass_vis:
            self.target(*in_vars, vis=vis)
        else:
            self.target(*in_vars)

        if self.global_dict is not None:
            self.global_dict['visualize'] = False
