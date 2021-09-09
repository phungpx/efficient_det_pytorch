import os
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter as _SummaryWriter


class SummaryWriter(_SummaryWriter):
    def __init__(self, log_dir: str = None, **kwargs):
        if log_dir is not None:
            log_dir = os.path.join(log_dir, datetime.now().strftime('%y%m%d%H%M'))

        super(SummaryWriter, self).__init__(log_dir, **kwargs)

    def state_dict(self) -> Any:
        return OrderedDict({
            'log_dir': self.log_dir,
            'purge_step': self.purge_step,
            'max_queue': self.max_queue,
            'flush_secs': self.flush_secs,
            'filename_suffix': self.filename_suffix,
        })

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        backup_log_dir = Path(state_dict.pop('log_dir'))

        for log in backup_log_dir.glob('*'):
            shutil.copy(log, self.log_dir)

        for name, attrib in state_dict.items():
            setattr(self, name, attrib)

        self.file_writer = self.all_writers = None
        self._get_file_writer()
