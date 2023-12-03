import os

from evaluator import Evaluator
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
    evaluator = Evaluator()

    # Actual evaluation
    eval_doctor = evaluator.execute(model, "a doctor", 5)
    eval_nurse = evaluator.execute(model, "a nurse", 5)

    p = 0


def evaluate_medicine():
    pass


if __name__ == "__main__":
    main()
