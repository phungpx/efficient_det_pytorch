from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter as _SummaryWriter


class TensorBoard:
    def __init__(self, logdir: str = None, **kwargs):
        super(TensorBoard, self).__init__()
        if logdir is not None:
            logdir = Path(logdir) / datetime.now().strftime('%y%m%d%H%M') / 'tensor_board'

        self.writer = _SummaryWriter(str(logdir), **kwargs)

    def add_scalar(self, tag, scalar_value, global_step):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict=tag_scalar_dict, global_step=global_step)
