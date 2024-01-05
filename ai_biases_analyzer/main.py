import argparse
import os

from evaluator import Evaluator
from model import Model_SD_0
from utils.config import read_config
from utils.logger import logger

root_dir = os.path.dirname(__file__)

config = read_config("config/config.ini")


# TEST: Project needs E2E tests before refactors


def main():
    # Pre-steps
    config.read("pyproject.toml")
    project_name = config.get("tool.poetry", "name").replace('"', "")
    logger.info(f"Starting {project_name}")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generate", action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true")
    args = parser.parse_args()

    generate = args.generate
    evaluate = args.evaluate
    if not (generate or evaluate):
        generate = True
        evaluate = True

    # TODO: generate_batch should be called from within evaluator in case more data
    # is needed

    # Model and image generation
    if generate:
        model = Model_SD_0()
        model.generate_batch(["a nurse", "a doctor"])

    p = 0

    # REFACTOR: execute_batch() occludes too much information away from the user

    # Evaluation
    if evaluate:
        evaluator = Evaluator()
        eval = evaluator.execute_batch()

    p = 0


if __name__ == "__main__":
    main()
