import yaml
import __main__

from importlib import import_module


def load_yaml(yaml_file):
    with open(yaml_file) as f:
        settings = yaml.safe_load(f)
    return settings


def eval_config(config):
    if isinstance(config, dict):
        for key, value in config.items():
            if key not in ['module', 'class']:
                config[key] = eval_config(value)

        if 'module' in config and 'class' in config:
            module = config['module']
            class_ = config['class']
            config_kwargs = config.get(class_, {})
            return getattr(import_module(module), class_)(**config_kwargs)

        return config
    elif isinstance(config, list):
        return [eval_config(ele) for ele in config]
    elif isinstance(config, str):
        return eval(config, __main__.__extralibs__)
    else:
        return config
