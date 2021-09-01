import torch

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

    def __init__(self, dataset, device, max_epochs=1):
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
    def __init__(self, dataset, device, max_norm, norm_type=2, max_epochs=1):
        super(Trainer, self).__init__(dataset, device, max_epochs)
        self.max_norm = max_norm
        self.norm_type = norm_type

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
        cls_preds, reg_preds, anchors = self.model._forward(samples)
        cls_loss, reg_loss = self.loss(cls_preds, reg_preds, anchors, targets)
        loss = cls_loss.mean() + reg_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm, self.norm_type)
        self.optimizer.step()

        return loss.item()


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

            cls_preds, reg_preds, anchors = self.model._forward(samples)

            return cls_preds, reg_preds, anchors, targets, image_infos
