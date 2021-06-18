import time

from ..module import Module
from ignite.engine import Events


class ScreenLogger(Module):
    def __init__(self, eval_names=None):
        super(ScreenLogger, self).__init__()
        self.eval_names = eval_names if eval_names else []

    def _started(self, engine):
        print(f'{time.asctime()} - STARTED')

    def _completed(self, engine):
        print(f'{time.asctime()} - COMPLETED')

    def _log_screen(self, engine):
        msg = f'Epoch #{engine.state.epoch} - {time.asctime()} - '
        for eval_name in self.eval_names:
            for metric_name, metric_value in self.frame['metrics'].metric_values[eval_name].items():
                msg += f'{eval_name}_{metric_name}: {metric_value:.5f} - '
        print(msg[:-2])

    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.frame['engine'].engine.add_event_handler(Events.STARTED, self._started)
        self.frame['engine'].engine.add_event_handler(Events.COMPLETED, self._completed)

        if len(self.eval_names):
            assert 'metrics' in self.frame, 'The frame does not have metrics.'
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self._log_screen)
