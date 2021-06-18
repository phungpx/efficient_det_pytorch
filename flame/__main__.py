import sys

from . import utils
from .module import Module
from importlib import import_module


class Frame(dict):
    def __init__(self, config_path):
        super(Frame, self).__init__()
        self.config_path = config_path


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = utils.load_yaml(config_path)
    frame = Frame(config_path)

    __extralibs__ = {name: import_module(lib) for (name, lib) in config.pop('extralibs', {}).items()}
    __extralibs__['config'] = config

    config = utils.eval_config(config)

    for module_name, module in config.items():
        if isinstance(module, Module):
            module.attach(frame, module_name)
        else:
            frame[module_name] = module

    for module in frame.values():
        if isinstance(module, Module):
            module.init()

    assert 'engine' in frame, 'The frame does not have engine.'
    frame['engine'].run()
