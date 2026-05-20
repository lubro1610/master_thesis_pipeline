"""Generate Appendix J figure: estimated latent policy stance."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

PNG_OUT = os.path.join(OUT_DIR, 'figure_appendix_j_latent_index.png')
PDF_OUT = os.path.join(OUT_DIR, 'figure_appendix_j_latent_index.pdf')


CUT_COLOR = '#b04a4a'
HOLD_COLOR = '#555555'
HIKE_COLOR = '#4f7f4f'
PHI_COLOR = '#3d6e8e'


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
        'legend.fontsize': 9,
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


def compute_latent_index(result, X):
    """Return latent index β'x_t for each meeting."""
    params = result.params
    tau_1, tau_2 = result.model.transform_threshold_params(params.values)[1:-1]
    beta = params.drop(['-1/0', '0/1'])
    bxt = X.values @ beta.values
    return bxt, tau_1, tau_2


def build_figure(bxt, y, tau_1, tau_2):
    """Create the latent-index and density-weight panels."""
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(9, 7.5), sharex=True,
        gridspec_kw={'height_ratios': [2.2, 1.0]},
    )

    bins = np.linspace(min(bxt.min(), tau_1) - 0.3, max(bxt.max(), tau_2) + 0.3, 28)

    bxt_cut = bxt[y == -1]
    bxt_hold = bxt[y == 0]
    bxt_hike = bxt[y == 1]

    ax_a.hist(
        [bxt_cut, bxt_hold, bxt_hike], bins=bins, stacked=True,
        color=[CUT_COLOR, HOLD_COLOR, HIKE_COLOR],
        label=[f'Cut (n={len(bxt_cut)})',
               f'Hold (n={len(bxt_hold)})',
               f'Hike (n={len(bxt_hike)})'],
        edgecolor='white', linewidth=0.4, alpha=0.92,
    )

    # Fixed limit keeps the panel stable across output formats.
    ax_a.set_ylim(0, 20)

    ax_a.axvline(tau_1, color='black', linestyle='--', linewidth=1.0, alpha=0.85)
    ax_a.axvline(tau_2, color='black', linestyle='--', linewidth=1.0, alpha=0.85)

    y_top = ax_a.get_ylim()[1]
    ax_a.text(tau_1, y_top * 0.96, fr'$\tau_1 = {tau_1:.2f}$',
              ha='center', va='top', fontsize=9,
              bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.85))
    ax_a.text(tau_2, y_top * 0.96, fr'$\tau_2 = {tau_2:.2f}$',
              ha='center', va='top', fontsize=9,
              bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.85))

    ax_a.set_ylabel('Number of meetings')
    ax_a.legend(loc='upper left', frameon=False)
    ax_a.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.6)
    ax_a.set_axisbelow(True)

    grid = np.linspace(bins[0], bins[-1], 400)
    phi_curve = norm.pdf(tau_2 - grid)
    ax_b.plot(grid, phi_curve, color=PHI_COLOR, linewidth=1.6,
              label=r'$\phi(\tau_2 - \beta^{\prime}x_t)$')

    phi_obs = norm.pdf(tau_2 - bxt)
    ax_b.scatter(bxt, phi_obs, s=14, color=PHI_COLOR, alpha=0.55,
                 edgecolor='white', linewidth=0.3)

    ax_b.axvline(tau_1, color='black', linestyle='--', linewidth=1.0, alpha=0.85)
    ax_b.axvline(tau_2, color='black', linestyle='--', linewidth=1.0, alpha=0.85)

    ax_b.set_xlabel(r'Estimated latent policy stance, $\beta^{\prime}x_t$')
    ax_b.set_ylabel(r'Density weight $\phi(\cdot)$')
    ax_b.legend(loc='upper left', frameon=False)
    ax_b.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.6)
    ax_b.set_axisbelow(True)
    ax_b.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def main():
    set_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    result, X, y = estimate_model()
    bxt, tau_1, tau_2 = compute_latent_index(result, X)

    print(f'n={len(y)}')
    print(f'cutpoints: tau_1={tau_1:.4f}, tau_2={tau_2:.4f}')
    print(f'latent_index: min={bxt.min():.3f}, mean={bxt.mean():.3f}, max={bxt.max():.3f}')
    print(f'decisions: cut={(y == -1).sum()}, hold={(y == 0).sum()}, hike={(y == 1).sum()}')

    n = len(bxt)
    n_below_tau1 = (bxt < tau_1).sum()
    n_between = ((bxt >= tau_1) & (bxt < tau_2)).sum()
    n_above_tau2 = (bxt >= tau_2).sum()
    print(f'below_tau_1: {n_below_tau1} ({n_below_tau1 / n * 100:.1f}%)')
    print(f'between_cutpoints: {n_between} ({n_between / n * 100:.1f}%)')
    print(f'above_tau_2: {n_above_tau2} ({n_above_tau2 / n * 100:.1f}%)')

    fig = build_figure(bxt, y.values, tau_1, tau_2)

    fig.savefig(PNG_OUT, dpi=600)
    fig.savefig(PDF_OUT)
    plt.close(fig)

    print(f'PNG: {PNG_OUT}')
    print(f'PDF: {PDF_OUT}')


if __name__ == '__main__':
    main()
