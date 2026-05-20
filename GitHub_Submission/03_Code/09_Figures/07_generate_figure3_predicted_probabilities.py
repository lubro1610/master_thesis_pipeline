"""Generate Figure 3: average predicted decision probabilities."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))

MASTER_CSV = os.path.join(WORKSPACE, '02_Data', 'master_dataset_with_sentiment.csv')
OUT_DIR = os.path.join(WORKSPACE, '04_Output', 'figures')

PNG_OUT = os.path.join(OUT_DIR, 'figure3_predicted_probabilities.png')
PDF_OUT = os.path.join(OUT_DIR, 'figure3_predicted_probabilities.pdf')


CUT_COLOR = '#b04a4a'
HOLD_COLOR = '#555555'
HIKE_COLOR = '#4f7f4f'


def set_plot_style():
    """Apply a restrained thesis-friendly visual style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#555555',
        'axes.linewidth': 0.8,
        'grid.color': '#e0e0e0',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.9,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def estimate_model():
    """Load data and estimate Model 3 (FinBERT ordered probit)."""
    df = pd.read_csv(MASTER_CSV)
    df['meeting_date'] = pd.to_datetime(df['meeting_date'])

    key_vars = [
        'decision_ordinal', 'inflation_gap', 'output_next',
        'lagged_rate', 'finbert_sentiment_std',
    ]
    df_clean = df[key_vars].dropna()
    y = df_clean['decision_ordinal']
    X = df_clean[['inflation_gap', 'output_next', 'lagged_rate', 'finbert_sentiment_std']]

    model = OrderedModel(y, X, distr='probit')
    result = model.fit(method='bfgs', disp=False)

    return result, X, y


def compute_predicted_probs(result, X):
    """Average predicted probabilities over observed covariate values."""
    params = result.params
    threshold_0, threshold_1 = result.model.transform_threshold_params(params.values)[1:-1]
    beta = params.drop(['-1/0', '0/1'])

    sentiment_grid = np.linspace(-3, 3, 500)
    p_cut = np.empty_like(sentiment_grid)
    p_hold = np.empty_like(sentiment_grid)
    p_hike = np.empty_like(sentiment_grid)

    X_obs = X.copy()
    sent_idx = X.columns.get_loc('finbert_sentiment_std')
    X_vals = X_obs.values.copy()
    beta_vals = beta.values

    for s_idx, s in enumerate(sentiment_grid):
        X_vals[:, sent_idx] = s
        bxt = X_vals @ beta_vals

        Phi_0 = norm.cdf(threshold_0 - bxt)
        Phi_1 = norm.cdf(threshold_1 - bxt)

        p_cut[s_idx]  = Phi_0.mean()
        p_hold[s_idx] = (Phi_1 - Phi_0).mean()
        p_hike[s_idx] = (1 - Phi_1).mean()

    return sentiment_grid, p_cut, p_hold, p_hike


def build_figure(sentiment_grid, p_cut, p_hold, p_hike):
    """Create the predicted-probability figure."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(sentiment_grid, p_cut,  color=CUT_COLOR,  linewidth=2.0, label='P(Cut)')
    ax.plot(sentiment_grid, p_hold, color=HOLD_COLOR,  linewidth=2.0, label='P(Hold)')
    ax.plot(sentiment_grid, p_hike, color=HIKE_COLOR,  linewidth=2.0, label='P(Hike)')

    # Mean sentiment is zero after standardization.
    ax.axvline(0, color='#b0b0b0', linewidth=0.8, linestyle=':')

    # Mark +/- 1 standard deviation for comparison with AME results.
    for s in [-1, 1]:
        ax.axvline(s, color='#cccccc', linewidth=0.6, linestyle='--', zorder=0)
    ax.text(-1, 0.97, '$-1$ SD', ha='center', va='top', fontsize=8.5, color='#888888')
    ax.text( 1, 0.97, '$+1$ SD', ha='center', va='top', fontsize=8.5, color='#888888')

    ax.set_xlabel('FinBERT sentiment (standard deviations from mean)')
    ax.set_ylabel('Average predicted probability')
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    ax.grid(True, axis='y', linewidth=0.6)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    ax.tick_params(colors='#333333')

    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.98, 0.88))

    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_plot_style()

    result, X, y = estimate_model()

    params = result.params
    beta_fb = params['finbert_sentiment_std']
    p_fb = result.pvalues['finbert_sentiment_std']
    threshold_0, threshold_1 = result.model.transform_threshold_params(params.values)[1:-1]
    print(f'n={len(y)}')
    print(f'finbert_sentiment_std: beta={beta_fb:.4f}, p={p_fb:.4f}')
    print(f'cutpoints: tau1={threshold_0:.4f}, tau2={threshold_1:.4f}')

    sentiment_grid, p_cut, p_hold, p_hike = compute_predicted_probs(result, X)

    idx_mean = np.argmin(np.abs(sentiment_grid))
    print(f'std=0: P(Cut)={p_cut[idx_mean]:.1%}, P(Hold)={p_hold[idx_mean]:.1%}, P(Hike)={p_hike[idx_mean]:.1%}')
    idx_plus1 = np.argmin(np.abs(sentiment_grid - 1))
    print(f'std=+1: P(Cut)={p_cut[idx_plus1]:.1%}, P(Hold)={p_hold[idx_plus1]:.1%}, P(Hike)={p_hike[idx_plus1]:.1%}')
    idx_minus1 = np.argmin(np.abs(sentiment_grid + 1))
    print(f'std=-1: P(Cut)={p_cut[idx_minus1]:.1%}, P(Hold)={p_hold[idx_minus1]:.1%}, P(Hike)={p_hike[idx_minus1]:.1%}')

    fig = build_figure(sentiment_grid, p_cut, p_hold, p_hike)
    fig.savefig(PNG_OUT, dpi=600)
    fig.savefig(PDF_OUT)
    plt.close(fig)

    print(f'PNG: {PNG_OUT}')
    print(f'PDF: {PDF_OUT}')


if __name__ == '__main__':
    main()
