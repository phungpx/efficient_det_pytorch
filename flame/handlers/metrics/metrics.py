from ...module import Module
from ignite.engine import Events


class Metrics(Module):
    def __init__(self, metrics, attach_to=None):
        super(Metrics, self).__init__()
        self.metrics = metrics
        self.metric_values = {}
        self.attach_to = attach_to if attach_to else {}

    def init(self):
        assert all(map(lambda x: x in self.frame, self.attach_to.keys())), f'The frame does not have all {self.attach_to.keys()}.'
        for evaluator, eval_name in self.attach_to.items():
            evaluator = self.frame[evaluator]

            for metric_name, metric in self.metrics.items():
                metric.attach(evaluator.engine, metric_name)

            evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, self._save_eval_result, eval_name)

    def _save_eval_result(self, engine, eval_name):
        self.metric_values[eval_name] = engine.state.metrics
