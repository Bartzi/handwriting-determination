import configparser

import chainer
import numpy


def parse_config(file_name, args):
    config = configparser.ConfigParser()
    config.read(file_name)

    for key in config['PATHS']:
        value = config['PATHS'][key]
        if len(value) == 0:
            value = None
        setattr(args, key, value)

    for key in config["SIZES"]:
        values = [int(v.strip()) for v in config['SIZES'][key].split(',')]
        setattr(args, key, values)

    for section, conversion_func in zip(["HYPERPARAMETERS_FLOAT", "HYPERPARAMETERS_INT"], [float, int]):
        for key in config[section]:
            setattr(args, key, conversion_func(config[section][key]))

    for key in config["TRAIN_PARAMS"]:
        setattr(args, key, int(config["TRAIN_PARAMS"][key]))

    for key in config["DTYPE"]:
        setattr(args, key, config["DTYPE"][key])

    for key in config["MISC"]:
        setattr(args, key, config["MISC"][key])

    return args
