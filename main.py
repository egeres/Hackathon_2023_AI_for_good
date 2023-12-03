import os

from batch_evaluator import BatchEvaluator
from model import Model_SD_0
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
    model = Model_SD_0()
    evaluator = BatchEvaluator()

    # Actual evaluation
    eval = evaluator.execute()


if __name__ == "__main__":
    main()
