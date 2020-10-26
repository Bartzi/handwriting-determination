import numpy
from chainer import backend
from chainer.training.extensions import Evaluator


class TensorboardEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):
        self.tensorboard_handle = kwargs.pop('tensorboard_handle')
        self.base_key = kwargs.pop('base_key', 'eval')
        super().__init__(*args, **kwargs)

    def __call__(self, trainer=None):
        summary = super().__call__(trainer=trainer)
        self.log_summary(trainer, summary)
        return summary

    def log_summary(self, trainer, summary):
        for key, value in summary.items():
            cpu_value = numpy.empty(value.shape, dtype=value.dtype)
            backend.copyto(cpu_value, value)
            self.tensorboard_handle.add_scalar('/'.join([self.base_key, key]), cpu_value, trainer.updater.iteration if trainer is not None else None)
