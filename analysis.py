import matplotlib.colors as mcolors
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def interpolate_color(color1, color2, factor: float) -> str:
    """Interpolate between two hex colors."""
    rgb1 = mcolors.hex2color(color1)
    rgb2 = mcolors.hex2color(color2)
    mixed_rgb = [rgb1[i] + factor * (rgb2[i] - rgb1[i]) for i in range(3)]
    return mcolors.to_hex(mixed_rgb)


def analysis_chart_genderrace(input_data: list[dict]) -> go.Figure:
    print("[green]Plotting...")

    """input_data looks like:
    [
        {
            "prompt":"a doctor",
            "prob_gender_Woman":0.2,
            "prob_gender_Man":0.8,
        },
    ]

    This method returns a plot with dots for each prompt, where the x-axis is
    the probability of gender. Hovering over the dots displays the prompt used
    """

    x_data, y_data, hover_text, colors = [], [], [], []

    for item in input_data:
        # X-axis: Probability of being male
        x_data.append(item["prob_gender_Man"])
        # Y-axis: Arbitrary, can be a fixed value as we don't have a y-axis metric
        y_data.append(1)
        # Hover text: The prompt
        hover_text.append(item["prompt"])
        # Color interpolation based on male probability
        color = interpolate_color("#fa82e2", "#7fb9fa", item["prob_gender_Man"])
        colors.append(color)

    fig = go.Figure(
        data=go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            text=hover_text,
            hoverinfo="text",
            # Set marker size and color
            marker=dict(
                color=colors,
                size=18,
                # line=dict(width=2, color="rgba(0, 0, 0, 0.5)"),
                opacity=0.3,
            ),
        ),
    )

    fig.update_layout(
        title="Gender Probability Analysis",
        xaxis_title="Probability of Gender Being Male",
        yaxis_title="",
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )

    return fig
