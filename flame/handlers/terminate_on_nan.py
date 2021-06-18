import ignite

from ..module import Module
from ignite.engine import Events


class TerminateOnNan(ignite.handlers.TerminateOnNan, Module):
    def init(self):
        self.frame['engine'].engine.add_event_handler(Events.ITERATION_COMPLETED, self)
