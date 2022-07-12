from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from typing import Dict


class Loss(Metric):
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(Loss, self).__init__(output_transform)
        self._loss_fn = loss_fn

    def reset(self):
        self._loss = 0
        self._cls_loss = 0
        self._reg_loss = 0
        self._num_examples = 0
        self.iter_metric = {}

    def update(self, output) -> Dict[str, float]:
        cls_loss, reg_loss = self._loss_fn(*output)
        loss = cls_loss.mean() + reg_loss.mean()

        if len(loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        loss = loss.item()
        cls_loss = cls_loss.mean().item()
        reg_loss = reg_loss.mean().item()

        N = output[0].shape[0]
        self._loss += loss * N
        self._cls_loss += cls_loss * N
        self._reg_loss += reg_loss * N
        self._num_examples += N

    def compute(self) -> Dict[str, float]:
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed.')

        loss = self._loss / self._num_examples
        cls_loss = self._cls_loss / self._num_examples
        reg_loss = self._reg_loss / self._num_examples

        return {'focal_loss': loss, 'cls_loss': cls_loss, 'reg_loss': reg_loss}
