import argparse
import os
import time

from rich import print

from analysis import analysis_chart_genderrace
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

    # Model and image generation
    if generate:
        model = Model_SD_0()
        images = model.generate_batch()

    # Evaluation
    if evaluate:
        evaluator = Evaluator()
        eval = evaluator.execute_batch()


if __name__ == "__main__":
    main()
