import ignite

from ..module import Module
from ignite import engine
from ignite.engine import Events


class EarlyStopping(ignite.handlers.EarlyStopping, Module):
    '''
        A handler can be used to stop the training if no improvement after a given number of events.
        Parameters:
            score_name (str): name of a metric attached to the engine what defines how good training process is.
            mode (str): one of min, max. In min mode, running process will be stopped when the quantity monitored has stopped decreasing; in max mode it will be stopped when the quantity monitored has stopped increasing.
        See Ignite EarlyStopping for more details about other parameters.
    '''

    def __init__(self, patience, score_name, evaluator_name, mode='max'):
        if mode not in ['min', 'max']:
            raise ValueError(f'mode must be min or max. mode value found is {mode}')
        super(EarlyStopping, self).__init__(patience, score_function=lambda e: e.state.metrics[score_name] if mode == 'max' else - e.state.metrics[score_name], trainer=engine.Engine(lambda engine, batch: None))
        self.evaluator_name = evaluator_name

    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.trainer = self.frame['engine'].engine
        self.frame[self.evaluator_name].engine.add_event_handler(Events.EPOCH_COMPLETED, self)

    def state_dict(self):
        return {
            'best_score': self.best_score,
            'counter': self.counter,
        }

    def load_state_dict(self, state_dict):
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']
