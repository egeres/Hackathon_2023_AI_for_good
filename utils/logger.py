import os
import logging
import datetime

from utils.config import read_config

config_section = 'LOGGER'
config_file = 'config/config.ini'
datetime_format = '%Y-%m-%d %H-%M-%S'
root_dir = os.path.dirname(os.path.dirname(__file__))


def console_handler(format: str = None) -> logging.Handler:
    """
    Creates a log to console handler.
    :param format: log format specified in config
    :return: logging.Handler object
    """
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    if format is not None:
        formatter = logging.Formatter(format)
        console_handler.setFormatter(formatter)
    return console_handler


def file_handler(file_path: str, format: str = None) -> logging.Handler:
    """
    Creates a log to file handler
    :param file_path: File path to write log to
    :param format: Log format
    :return: logging.Handler object
    """
    log_dir = os.path.dirname(file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.isfile(file_path):
        open(file_path, 'a').close()

    handler = logging.FileHandler(file_path, 'a')
    handler.setLevel(logging.DEBUG)
    if format is not None:
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
    return handler


config = read_config(config_file)
log_file_path = config.get(config_section, 'log_file_path')
file_format = config.get(config_section, 'log_file_format')
console_format = config.get(config_section, 'log_console_format')

if '{' in log_file_path and '}' in log_file_path:
    log_file_path = log_file_path.format(datetime.datetime.now().strftime(datetime_format))

logger = logging.getLogger()
logger.addHandler(console_handler(format=console_format))
if log_file_path is not None and log_file_path != '':
    logger.addHandler(file_handler(log_file_path))
