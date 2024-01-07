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
    """Test meant to ensure that the end-to-end pipeline doesn't crash for this plot"""
    # REFACTOR: This test is not e2e, generator is not tested.
    evaluator = Evaluator()
    eval = evaluator.execute_batch()
    assert isinstance(eval, list)
    plot_chart_genderrace(eval)
