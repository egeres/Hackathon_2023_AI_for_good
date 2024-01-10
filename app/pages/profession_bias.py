import streamlit as st

from ai_biases_analyzer.evaluator import Evaluator
from ai_biases_analyzer.model import Model_SD_0
from ai_biases_analyzer.plot import plot_chart_genderrace

st.markdown("# Profession Bias")
st.sidebar.markdown("# Profession Bias")


evaluator = Evaluator()
eval = evaluator.execute_batch()


st.plotly_chart(
    plot_chart_genderrace(eval),
    use_container_width=True,
)
