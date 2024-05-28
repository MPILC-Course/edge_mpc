import plotly.graph_objects as go
from plotly_resampler import FigureWidgetResampler
import matplotlib.pyplot as plt


def plot_features_and_target(df, features):
    fig = FigureWidgetResampler(go.Figure())
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    # Add a trace for each feature
    for feature in features:
        fig.add_trace(
            go.Scattergl(name=f'{feature}', showlegend=True),
            hf_x=df.index,
            hf_y=df[feature]
        )

    fig.update_layout(height=400, template="plotly_dark")
    return fig

def histogram(y, bins):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    # Histogram of residuals
    axs.hist(y, bins=bins, color='black', edgecolor='gray')
    axs.set_xlabel(r'$t_{wall}$ [s]')
    axs.set_ylabel('Samples')
    axs.set_yscale('log')
    axs.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Adjust layout for better presentation
    plt.tight_layout()
    plt.savefig('img/twall.eps', format='eps', transparent=True)
    
    plt.close()
    


    return fig