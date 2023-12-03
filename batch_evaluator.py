import os
import pickle
from multiprocessing.pool import ThreadPool

import pandas as pd

from evaluator import Evaluator
from model import hex_hash
from utils.config import read_config
from utils.logger import logger

config = read_config('config/config.ini')
def evaluate(hex_prompt, prompt):
    img_path = config.get('EVALUATOR', 'image_folder')
    available_prompts = os.listdir(img_path)
    if hex_prompt in available_prompts:
        paths = [os.path.join(img_path, prompt, p)
                 for p in os.listdir(os.path.join(img_path, prompt))
                 if p.endswith('.png')]
        evaluator = Evaluator()
        res = evaluator.execute(prompt, paths)
        return res


class BatchEvaluator:
    def execute(self):
        logger.info('Batch processing images')

        # Pre-Execution
        img_path = config.get('EVALUATOR', 'image_folder')
        prompts = config.get('EVALUATOR', 'prompts').split(',')
        hex_prompts = [hex_hash(x) for x in prompts]
        available_prompts = os.listdir(img_path)
        n_threads = config.get('EVALUATOR', 'n_threads')

        # Execute in multi-thread
        res = []
        for i, prompt in enumerate(hex_prompts):
            if prompt in available_prompts:
                paths = [os.path.join(img_path, prompt, p)
                         for p in os.listdir(os.path.join(img_path, prompt))
                         if p.endswith('.png')]
                evaluator = Evaluator()
                res.append(evaluator.execute(prompts[i], paths))
        return res