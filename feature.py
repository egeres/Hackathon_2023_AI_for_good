from deepface import DeepFace

from utils.logger import logger
from utils.config import read_config

config = read_config('config/config.ini')


def get_features(img_path, features=None):
    logger.debug("Evaluating image {}".format(img_path))
    features = DeepFace.analyze(img_path=img_path, actions=features)
    return features


def get_features_batch(inputs):
    logger.debug(f"Starting to batch analyze {len(inputs)} images, this may take a while.")
    res = []
    for img in inputs:
        try:
            features = get_features(img, features=['gender', 'race'])
            res.extend(features)
        except ValueError:
            logger.warning(f"No faces were detected in {img}")
    logger.info(f"Extracted features for {len(res)} faces")
    return res

def analyze_features()
