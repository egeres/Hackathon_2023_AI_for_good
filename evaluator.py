import os
from math import log2

import pandas as pd

from feature import get_features_batch
from model import Model
from utils.config import read_config
from utils.logger import logger

config = read_config("config/config.ini")


class Evaluator:
    def __init__(self):
        self.features = config.get("EVALUATOR", "features").split(",")

    def execute(self, model: Model, prompt: str, n_images: int) -> dict:
        logger.info("Starting Evaluator Execution")

        pictures_paths = model.generate(
            prompt,
            number_of_imgs=n_images,
        )
        # pictures_paths = [
        #     r"C:\Github\Hackathon_2023_AI_for_good\outputs\2023-12-03_00-30-33_a doctor.png",
        #     r"C:\Github\Hackathon_2023_AI_for_good\outputs\2023-12-03_00-33-23_a doctor.png",
        #     r"C:\Github\Hackathon_2023_AI_for_good\outputs\2023-12-03_00-42-06_a doctor.png",
        # ]

        # Old implementation
        # faces = get_features_batch(outputs, self.features)
        # self.subfeatures = self._get_subfeatures(faces[0])
        # result = self.analyze_features(prompt, faces)
        # return result

        pictures_analysis: list[dict] = get_features_batch(
            pictures_paths,
            self.features,
        )
        df = pd.DataFrame(pictures_analysis)
        result = {
            "prompt": prompt,
            "gender_analysis": self.get_probabilities_1(df, "gender"),
            "race_analysis": self.get_probabilities_1(df, "dominant_race"),
        }

        return result

    def get_probabilities_1(self, df: pd.DataFrame, column: str) -> dict:
        """Given the name of a column, finds the unique values in that column and
        computes their probabilities. Returns a dict with the probabilities."""

        assert column in df.columns, f"Column {column} not in df.columns"

        probabilities = {}
        for value in df[column].unique():
            p = len(df[df[column] == value]) / len(df)
            probabilities[value] = p
        return probabilities

    def analyze_features(self, prompt: str, features: list) -> dict:
        logger.info("Starting to analyze features")
        # Convert features to df
        df_features = pd.DataFrame(features)
        df_features = df_features[
            ["source", "region"]
            + ["dominant_" + f for f in self.features]
            + self.features
        ]
        df_features = self._unpack_columns(df_features, self.features + ["region"])

        # Compute distribution of possible biases by feature
        logger.debug("Computing probabilities")
        probabilities = self._compute_probabilities(df_features)
        average_representation = self._compute_average_representation(df_features)
        entropy = self._compute_entropy(df_features)
        representation_entropy = self._compute_representation_entropy(df_features)

        result = {"prompt": prompt}
        result.update(entropy)
        result.update(representation_entropy)
        result.update(probabilities)
        result.update(average_representation)
        logger.debug(result)
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
                col_name = "dominant_" + feature
                p = len(df_features[df_features[col_name] == subfeature]) / len(
                    df_features
                )
                probabilities["prob_" + feature + "_" + subfeature] = p
        return probabilities

    def _compute_average_representation(self, df_features: pd.DataFrame) -> dict:
        average_representation = {}
        for i, feature in enumerate(self.features):
            for subfeature in self.subfeatures[i]:
                col_name = feature + "_" + subfeature
                avg = (df_features[col_name] / 100).mean()
                average_representation[
                    "average_representation_" + feature + "_" + subfeature
                ] = avg
        return average_representation

    def _compute_entropy(self, df_features: pd.DataFrame) -> dict:
        entropy = {}
        for i, feature in enumerate(self.features):
            feature_entropy = 0.0
            col_name = "dominant_" + feature
            for subfeature in self.subfeatures[i]:
                p = len(df_features[df_features[col_name] == subfeature]) / len(
                    df_features
                )
                if p > 0.0:
                    feature_entropy -= p * log2(p)
            entropy["entropy_" + feature] = feature_entropy
        return entropy

    def _compute_representation_entropy(self, df_features: pd.DataFrame) -> dict:
        entropy = {}
        for i, feature in enumerate(self.features):
            feature_entropy = 0.0
            for subfeature in self.subfeatures[i]:
                col_name = feature + "_" + subfeature
                p = (df_features[col_name] / 100).mean()
                if p > 0.0:
                    feature_entropy -= p * log2(p)
            entropy["representation_entropy_" + feature] = feature_entropy
        return entropy

    @staticmethod
    def _unpack_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            unpacked_df = pd.json_normalize(df[col])
            for unpacked_col in unpacked_df.columns:
                unpacked_df = unpacked_df.rename(
                    columns={unpacked_col: col + "_" + unpacked_col}
                )
            df = pd.concat([df.drop(col, axis=1), unpacked_df], axis=1)
        return df
