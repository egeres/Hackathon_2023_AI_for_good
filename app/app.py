import numpy as np
import streamlit as st
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.markdown("# Home")
st.sidebar.markdown("# Home 🎈")

# Load data
# TODO: To substitute with load data method, possibly better to make evaluator save jsons and load them in here.
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

#
st.title('AI Image Generator Model Bias Analysis')

# Create distplot with custom bin_size
hist_data = [x1, x2, x3]
group_labels = ['White', 'Black', 'Asian']
fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)