import os

from evaluator import Evaluator
from model import Model
from utils.config import read_config
from utils.logger import logger

root_dir = os.path.dirname(__file__)
config = read_config("config/config.ini")


def main():
    # Pre-steps
    config.read("pyproject.toml")
    project_name = config.get("tool.poetry", "name").replace('"', "")
    logger.info(f"Starting {project_name}")

    # Model & evaluator instantiation
    model = Model()
    evaluator = Evaluator()

    # Actual evaluation
    eval = evaluator.execute(model, "a doctor", 10)


def evaluate_medicine():
    pass


if __name__ == "__main__":
    main()
