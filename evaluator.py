import os
from math import log2

import pandas as pd
from deepface import DeepFace

from model import Model
from feature import get_features_batch
from utils.logger import logger
from utils.config import read_config

config = read_config('config/config.ini')


class Evaluator:
    def __init__(self):
        self.features = config.get('EVALUATOR', 'features').split(',')

    def execute(self, model:Model, prompt: str, n_images: int) -> pd.DataFrame:
        logger.info('Starting Evaluator Execution')
        self.prompt = prompt
        #outputs = model.generate(prompt, n_images)
        # TODO: Merge with Rafa model generator
        outputs = ['data/img.png', 'data/img_1.png', 'data/img_2.png', 'data/img_3.png', 'data/img_4.png',
                   'data/img_5.png', 'data/img_6.png', 'data/img_7.png', 'data/img_8.png', 'data/img_9.png',
                   'data/img_10.png']

        faces = get_features_batch(outputs, self.features)

        self.subfeatures = self._get_subfeatures(faces[0])

        result = self.analyze_features(prompt, faces)
        return result

    def analyze_features(self, prompt: str, features: list) -> dict:
        logger.info('Starting to analyze features')
        # Convert features to df
        df_features = pd.DataFrame(features)
        df_features = df_features[['source', 'region'] + ['dominant_' + f for f in self.features] + self.features]
        df_features = self._unpack_columns(df_features, self.features + ['region'])


        probabilities = self._compute_probabilities(df_features)
        average_representation = self._compute_average_representation(df_features)
        entropy = self._compute_entropy(df_features)
        representation_entropy = self._compute_representation_entropy(df_features)

        result = {'prompt': prompt}
        result.update(entropy)
        result.update(representation_entropy)
        result.update(probabilities)
        result.update(average_representation)
        return result

    def _get_subfeatures(self, features: dict) -> list:
        subfeatures = []
        for f in self.features:
            subfeatures.append(list(features[f].keys()))
        return subfeatures

    def _compute_probabilities(self, df_features: pd.DataFrame) -> dict:
        probabilities = {}
        for i, feature in enumerate(self.features):
            for subfeature in self.subfeatures[i]:
                col_name = 'dominant_' + feature
                p = len(df_features[df_features[col_name] == subfeature]) / len(df_features)
                probabilities['prob_' + feature + '_' + subfeature] = p
        return probabilities

    def _compute_average_representation(self, df_features: pd.DataFrame) -> dict:
        average_representation = {}
        for i, feature in enumerate(self.features):
            for subfeature in self.subfeatures[i]:
                col_name = feature + '_' + subfeature
                avg = (df_features[col_name] / 100).mean()
                average_representation['average_representation_' + feature + '_' + subfeature] = avg
        return average_representation

    def _compute_entropy(self, df_features: pd.DataFrame) -> dict:
        entropy = {}
        for i, feature in enumerate(self.features):
            feature_entropy = .0
            col_name = 'dominant_' + feature
            for subfeature in self.subfeatures[i]:
                p = len(df_features[df_features[col_name] == subfeature])/len(df_features)
                if p > .0:
                    feature_entropy -= p * log2(p)
            entropy['entropy_' + feature] = feature_entropy
        return entropy

    def _compute_representation_entropy(self, df_features: pd.DataFrame) -> dict:
        entropy = {}
        for i, feature in enumerate(self.features):
            feature_entropy = .0
            for subfeature in self.subfeatures[i]:
                col_name = feature + '_' + subfeature
                p = (df_features[col_name]/100).mean()
                if p > .0:
                    feature_entropy -= p * log2(p)
            entropy['representation_entropy_' + feature] = feature_entropy
        return entropy

    @staticmethod
    def _unpack_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            unpacked_df = pd.json_normalize(df[col])
            for unpacked_col in unpacked_df.columns:
                unpacked_df = unpacked_df.rename(columns={unpacked_col: col + "_" + unpacked_col})
            df = pd.concat([df.drop(col, axis=1), unpacked_df], axis=1)
        return df


