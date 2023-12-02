import os
from configparser import ConfigParser


def read_config(config_file: str) -> ConfigParser:
    """
    Reads configuration file and returns ConfigParser object.
    :param root_dir: Root directory of project.
    :param config_file: config file path.
    :return: ConfigParser object.
    """
    config = ConfigParser()
    config_file = os.path.join(os.getcwd(), config_file)
    config.read(config_file)
    return config
