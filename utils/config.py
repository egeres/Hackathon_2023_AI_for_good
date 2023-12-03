import os
from configparser import ConfigParser


def read_config(config_file: str) -> ConfigParser:
    # TODO: Add root_dir
    """
    Reads configuration file and returns ConfigParser object.
    :param root_dir: Root directory of project.
    :param config_file: config file path.
    :return: ConfigParser object.
    """
    config = ConfigParser()
    config.read(os.path.join(os.getcwd(), config_file))
    return config
