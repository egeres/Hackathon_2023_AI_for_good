from configparser import ConfigParser
from pathlib import Path

# REFACTOR: This should use pathlib!!

# REFACTOR: Maybe the config should be a yaml, ini is so 90's


def read_config(config_file: str) -> ConfigParser:
    """
    Reads configuration file and returns ConfigParser object.
    :param root_dir: Root directory of project.
    :param config_file: config file path.
    :return: ConfigParser object.
    """
    config = ConfigParser()
    config.read(Path(__file__).parent.parent.parent / config_file)
    return config
