import torch

from ...module import Module
from ignite import engine as e
from abc import abstractmethod


class Evaluator(Module):
    '''
        Engine controls evaluating process.
        See Engine documentation for more details about parameters.
    '''
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
        super(Evaluator, self).__init__()
        self.device = device
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.engine = e.Engine(self._update)

    def run(self):
        return self.engine.run(self.dataset, self.max_epochs)

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        self.model = self.frame['model'].to(self.device)

    @abstractmethod
    def _update(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            samples = torch.stack([image.to(self.device) for image in params[0]], dim=0)
            targets = [{k: v.to(self.device) for k, v in target.items() if not isinstance(v, list)} for target in batch[1]]
            image_infos = [image_info for image_info in params[2]]

            predictions = self.model.inference(samples)

            return predictions, targets, image_infos
