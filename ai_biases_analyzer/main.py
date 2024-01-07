from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger
from utils.config import read_config

from ai_biases_analyzer.evaluator import Evaluator
from ai_biases_analyzer.model import Model, Model_SD_0

app = typer.Typer(help="AI Biases Analyzer")

root_dir = Path(__file__).parent.parent
config = read_config("config/config.ini")


@app.command(help="Run the end-to-end pipeline, generating images and evaluating them.")
def run(prompts: list[str] | None = None):
    generate(prompts)
    evaluate(prompts)


@app.command(help="Generate images from prompts.")
def generate(prompts: list[str] | None = None):
    if prompts is None:
        config.get("DEFAULT", "prompts").split(",")
    model = Model_SD_0()
    model.generate_batch(prompts)


@app.command(help="Evaluates images.")
def evaluate(prompts: list[str] | None = None):
    evaluator = Evaluator(prompts=prompts)
    eval = evaluator.execute_batch()
    logger.info(json.dumps(eval, indent=2))


if __name__ == "__main__":
    app()
