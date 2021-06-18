import os
import time
import torch
import shutil
import ignite

from ..module import Module
from pathlib import Path
from datetime import datetime
from ignite.engine import Events
from collections import defaultdict


class BestSaver(ignite.handlers.ModelCheckpoint, Module):
    '''
        A handler can be used to save the best objects according to a metric to disk.
        Parameters:
            dirname (str): directory path. A subfolder will be created in the directory. Your objects will be saved in this subfolder.
            score_name (str): name of a metric attached to the engine what defines how good objects are.
            evaluator_name (str): name of an engine module specified in config where saver is attached to.
            mode (str): one of min, max. In min mode, the least score objects will be saved. In max mode, the best score objects will be saved.
        See Ignite ModelCheckpoint for more details about other parameters.
    '''

    def __init__(self, dirname, score_name, evaluator_name, mode='max', n_saved=1, atomic=True, require_empty=True, create_dir=True):
        if mode not in ['min', 'max']:
            raise ValueError(f'mode must be min or max. mode value found is {mode}')

        dirname = os.path.join(dirname, datetime.now().strftime('%y%m%d%H%M'))

        super(BestSaver, self).__init__(dirname, 'best', score_function=lambda e: e.state.metrics[score_name] if mode == 'max' else - e.state.metrics[score_name], score_name=score_name, n_saved=n_saved, atomic=atomic, require_empty=require_empty, create_dir=create_dir, global_step_transform=lambda engine, event: self.frame['engine'].engine.state.epoch)
        self.evaluator_name = evaluator_name

    def init(self):
        assert self.evaluator_name in self.frame, f'The frame does not have {self.evaluator_name}.'
        assert 'model' in self.frame, 'The frame does not have model.'
        self.frame[self.evaluator_name].engine.add_event_handler(Events.EPOCH_COMPLETED, self, {'model': self.frame['model']})

    def state_dict(self):
        return {
            'n_saved': self.n_saved,
            'saved': [(priority, filename) for priority, filename in self._saved],
        }

    def load_state_dict(self, state_dict):
        self.n_saved = state_dict['n_saved']
        self._saved = [ignite.handlers.Checkpoint.Item(priority, filename) for priority, filename in state_dict['saved']]


class BackupSaver(ignite.handlers.ModelCheckpoint, Module):
    '''
        A handler can be used to periodically save objects to disk.
        Parameters:
            modules (list of str): names of modules that are backed up.
            dirname (str): directory path. A subfolder will be created in the directory. Your objects will be saved in this subfolder.
        See Ignite ModelCheckpoint for more details about other parameters.
    '''

    class Checkpoint(object):
        def __init__(self, modules, frame):
            super(BackupSaver.Checkpoint, self).__init__()
            assert all(map(lambda x: x in frame, modules)), f'The frame does not have all {modules}'

            if 'last_epoch' in modules:
                raise ValueError('modules should not have key last_epoch.')

            if 'last_iteration' in modules:
                raise ValueError('modules should not have key last_iteration.')

            self.frame = frame
            self.modules = modules

        def state_dict(self):
            checkpoint = {module: self.frame[module].state_dict() for module in self.modules}
            checkpoint['last_epoch'] = self.frame['engine'].engine.state.epoch
            checkpoint['last_iteration'] = self.frame['engine'].engine.state.iteration
            return checkpoint

    def __init__(self, modules, dirname, save_interval, n_saved=1, atomic=True, require_empty=True, create_dir=True):
        dirname = os.path.join(dirname, datetime.now().strftime('%y%m%d%H%M'))
        super(BackupSaver, self).__init__(dirname, 'backup', n_saved=n_saved, atomic=atomic, require_empty=require_empty, create_dir=create_dir, global_step_transform=lambda engine, event: engine.state.epoch)
        self.modules = modules
        self.save_interval = save_interval

    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        shutil.copy(self.frame.config_path, self.save_handler.dirname)
        checkpoint = self.Checkpoint(self.modules, self.frame)
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval), self, {'checkpoint': checkpoint})
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval), self._correct_checkpoint)

    def state_dict(self):
        return {
            'n_saved': self.n_saved,
            'saved': [(priority, filename) for priority, filename in self._saved],
        }

    def load_state_dict(self, state_dict):
        self.n_saved = state_dict['n_saved']
        self._saved = [ignite.handlers.Checkpoint.Item(priority, filename) for priority, filename in state_dict['saved']]

    def _correct_checkpoint(self, engine):
        checkpoint_path = self.frame[self.module_name].last_checkpoint
        checkpoint = torch.load(checkpoint_path)
        checkpoint[self.module_name]['saved'] = [(priority, filename) for priority, filename in self.frame[self.module_name]._saved]
        torch.save(checkpoint, checkpoint_path)


class CheckpointLoader(Module):
    def __init__(self, checkpoint_path, mode):
        super(CheckpointLoader, self).__init__()
        self.checkpoint_path = checkpoint_path
        if mode in ['train', 'resume', 'retrain', 'test']:
            self.mode = mode
        else:
            raise ValueError('mode must be train, resume, retrain or test.')

    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.frame['engine'].engine.add_event_handler(Events.STARTED, self._load_checkpoint, self.mode)

    def _load_checkpoint(self, engine, mode):
        if mode in ['resume']:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

            assert 'engine' in self.frame, 'The frame does not have engine.'
            self.frame['engine'].engine.state.epoch = checkpoint.pop('last_epoch')
            self.frame['engine'].engine.state.iteration = checkpoint.pop('last_iteration')

            for module, state_dict in checkpoint.items():
                self.frame[module].load_state_dict(state_dict)

                if isinstance(self.frame[module], ignite.handlers.ModelCheckpoint):
                    self._fake_saved(self.frame[module])
        elif mode in ['retrain', 'test']:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

            assert 'model' in self.frame, 'The frame does not have model.'
            self.frame['model'].load_state_dict(checkpoint)

    def _fake_saved(self, saver):
        for i, (priority, filename) in enumerate(saver._saved):
            fake_saved = Path(saver.save_handler.dirname).joinpath(filename)
            fake_saved.touch()
            saver._saved[i] = ignite.handlers.Checkpoint.Item(priority, filename)


class History(Module):
    def __init__(self):
        super(History, self).__init__()
        self.lr = []
        self.time = []
        self.epoch = []
        self.metric_values = defaultdict(lambda: defaultdict(list))

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'optim' in self.frame, 'The frame does not have optim.'
        assert 'engine' in self.frame, 'The frame does not have engine.'
        assert 'metrics' in self.frame, 'The frame does not have metrics.'
        self.model = self.frame['model']
        self.optim = self.frame['optim']
        self.engine = self.frame['engine']
        self.metrics = self.frame['metrics']
        self.engine.engine.add_event_handler(Events.EPOCH_COMPLETED, self._update)

    def _update(self, engine):
        self.lr.append(self.optim.state_dict()['param_groups'][0]['lr'])
        self.time.append(time.time())
        self.epoch.append(engine.state.epoch)
        for eval_name, metrics in self.metrics.metric_values.items():
            for metric_name, metric_value in metrics.items():
                self.metric_values[eval_name][metric_name].append(metric_value)

    def state_dict(self):
        return {
            'lr': self.lr,
            'time': self.time,
            'epoch': self.epoch,
            'metric_values': dict(self.metric_values),
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict['lr']
        self.time = state_dict['time']
        self.epoch = state_dict['epoch']
        self.metric_values = defaultdict(lambda: defaultdict(list), state_dict['metric_values'])
