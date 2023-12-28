from __future__ import annotations

from pathlib import Path

from deepface import DeepFace
from diskcache import FanoutCache

from utils.config import read_config
from utils.logger import logger

config = read_config("config/config.ini")


@FanoutCache("./.cache_features").memoize()
def get_features(img_path: str | Path, features: list) -> list[dict]:
    """Given the path to an image and a list of features to extract, returns a list of
    dictionaries representing the features of the detected faces on the image"""

    if isinstance(img_path, Path):
        img_path = str(img_path)
    # if isinstance(img_path, list):
    #     img_path = [str(path) for path in img_path]
    logger.debug(f"Evaluating image {img_path}")

    # Returns a list of dicts or a dict depending on 'img_path'!
    faces = DeepFace.analyze(img_path=img_path, actions=features)
    if isinstance(faces, dict):
        faces = [faces]
    for face in faces:
        face["source"] = img_path
    return faces


def get_features_batch(
    inputs: list[str | Path] | list[str] | list[Path], features: list
) -> list:
    logger.debug(f"Starting to batch analyze {len(inputs)} images...")
    res = []
    for img in inputs:
        try:
            result = get_features(img, features=features)
            res.extend(result)
        except ValueError:
            logger.warning(f"No faces were detected in {img}")
    logger.info(f"Extracted features for {len(res)} faces")
    return res
