from deepface import DeepFace

from utils.config import read_config
from utils.logger import logger

config = read_config("config/config.ini")


def get_features(img_path: str, features: list = None) -> list:
    logger.debug("Evaluating image {}".format(img_path))
    faces = DeepFace.analyze(img_path=img_path, actions=features)
    for face in faces:
        face["source"] = img_path
    return faces


def get_features_batch(inputs: list, features: list) -> list:
    logger.debug(
        f"Starting to batch analyze {len(inputs)} images, this may take a while."
    )
    res = []
    for img in inputs:
        try:
            result = get_features(img, features=features)
            res.extend(result)
        except ValueError:
            logger.warning(f"No faces were detected in {img}")
    logger.info(f"Extracted features for {len(res)} faces")
    return res
