from ..core.engine.engine import Evaluator
from ignite.engine import Events


class MetricEvaluator(Evaluator):
    def init(self):
        super(MetricEvaluator, self).init()
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self._run)

    def _run(self, engine):
        self.run()
