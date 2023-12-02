import os

from evaluator import Evaluator
from model import Model
from utils.logger import logger
from utils.config import read_config


root_dir = os.path.dirname(__file__)
config = read_config('config/config.ini')


def main():
    config.read('pyproject.toml')
    project_name = config.get('tool.poetry', 'name').replace('\"', '')
    logger.info(f"Starting {project_name}")
    model = Model()
    evaluator = Evaluator()
    eval = evaluator.execute(model, "a doctor", 10)

def evaluate_medicine():
    pass


if __name__ == '__main__':
    main()
