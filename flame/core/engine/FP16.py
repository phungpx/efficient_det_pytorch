import torch

from torch.cuda import amp
from typing import Optional
from ...module import Module
from ignite import engine as e
from abc import abstractmethod


class Engine(Module):
    '''
        Base class for all engines. Your engine should subclass this class.
        Class Engine contains an Ignite Engine that controls running process over a dataset.
        Method _update is a function receiving the running Ignite Engine and the current batch in each iteration and returns data to be stored in the Ignite Engine's state.
        Parameters:
            dataset_name (str): dataset which engine run over.
            device (str): device on which model and tensor is allocated.
            max_epochs (int): number of epochs training process runs.
    '''

    def __init__(self, dataset: str, device: str, max_epochs: int = 1):
        super(Engine, self).__init__()
        self.dataset = dataset
        self.device = device
        self.max_epochs = max_epochs
        self.engine = e.Engine(self._update)

    def run(self):
        return self.engine.run(self.dataset, self.max_epochs)

    @abstractmethod
    def _update(self, engine, batch):
        pass


class Trainer(Engine):
    def __init__(self, dataset: str, device: str, max_norm: Optional[float] = None, norm_type: int = 2, max_epochs: int = 1):
        super(Trainer, self).__init__(dataset, device, max_epochs)
        self.max_norm = max_norm
        self.norm_type = norm_type
        # self.scaler = amp.GradScaler()

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'optim' in self.frame, 'The frame does not have optim.'
        self.model = self.frame['model'].to(self.device)
        self.optimizer = self.frame['optim']
        self.loss = self.frame['loss']

    def _update(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
        samples = torch.stack([image.to(self.device) for image in params[0]], dim=0)
        targets = [{k: v.to(self.device) for k, v in target.items() if not isinstance(v, list)} for target in params[1]]

        with amp.autocast():
            cls_preds, reg_preds, anchors = self.model(samples)
            cls_loss, reg_loss = self.loss(cls_preds, reg_preds, anchors, targets)
            loss = cls_loss.mean() + reg_loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm, self.norm_type)
        self.optimizer.step()

        return loss.item()

        # self.scaler.scale(loss).backward()

        # # Unscales the gradients of optimizer's assigned params in-place
        # self.scaler.unscale_(self.optimizer)

        # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm, self.norm_type)

        # # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        # self.scaler.step(self.optimizer)

        # # Updates the scale for next iteration.
        # self.scaler.update()

        # return loss.item()


class Evaluator(Engine):
    '''
        Engine controls evaluating process.
        See Engine documentation for more details about parameters.
    '''

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        self.model = self.frame['model'].to(self.device)

    def _update(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            samples = torch.stack([image.to(self.device) for image in params[0]], dim=0)
            targets = [{k: v.to(self.device) for k, v in target.items() if not isinstance(v, list)} for target in batch[1]]
            image_infos = [image_info for image_info in params[2]]

            cls_preds, reg_preds, anchors = self.model(samples)

            return cls_preds, reg_preds, anchors, targets, image_infos
