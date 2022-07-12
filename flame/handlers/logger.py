import logging
from pathlib import Path
from datetime import datetime


class Logger:
    def __init__(
        self,
        logdir=None,
        logname=None,
        mode=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    ):
        logdir = Path(logdir) / datetime.now().strftime('%y%m%d%H%M') / 'log'
        if not logdir.exists():
            logdir.mkdir(parents=True)

        logpath = str(logdir / f'{logname}.log')

        self.logger = logging.getLogger(logname)
        self.logger.setLevel(mode)
        handler = logging.FileHandler(logpath)
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, message: str, verbose: bool = True):
        self.logger.info(message)
        if verbose:
            print(message)
