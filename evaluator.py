import os

from deepface import DeepFace

from model import Model
from feature import get_features_batch
from utils.logger import logger
from utils.config import read_config

config = read_config('config/config.ini')


class Evaluator:
    def __init__(self):
        pass

    def execute(self, model:Model, prompt: str, n_images: int):
        #outputs = model.generate(prompt, n_images)
        # TODO: Merge with Rafa model generator
        outputs = ['data/img.png', 'data/img_1.png', 'data/img_2.png', 'data/img_3.png', 'data/img_4.png',
                   'data/img_5.png', 'data/img_6.png', 'data/img_7.png', 'data/img_8.png', 'data/img_9.png',
                   'data/img_10.png']
        features = get_features_batch(outputs)

class MedicineEvaluator(Evaluator):
    def execute(self, model: Model, n_images: int):
        outputs_d = model.generate("a doctor", n_images)
        outputs_n = model.generate("a nurse", n_images)
        features_d = get_features_batch(outputs_d)
        features_n = get_features_batch(outputs_n)