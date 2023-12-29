# from biasdetector import Evaluator
from ai_biases_analyzer.evaluator import Evaluator
from ai_biases_analyzer.model import Model_SD_0
from ai_biases_analyzer.plot import plot_chart_genderrace


def test_instantiate_evaluator():
    """Instantiating evaluator shouldn't crash"""
    Evaluator()


def test_instantiate_model():
    """Instantiating model shouldn't crash"""
    Model_SD_0()


def test_e2e_0():
    evaluator = Evaluator()
    eval = evaluator.execute_batch()
    plot_chart_genderrace(eval)
