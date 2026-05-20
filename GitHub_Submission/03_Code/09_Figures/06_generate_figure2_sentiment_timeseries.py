"""Generate Figure 2: sentiment time series and policy rate."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))

MASTER_CSV = os.path.join(WORKSPACE, '02_Data', 'master_dataset_with_sentiment.csv')
OUT_DIR = os.path.join(WORKSPACE, '04_Output', 'figures')

PNG_OUT = os.path.join(OUT_DIR, 'figure2_sentiment_timeseries.png')
PDF_OUT = os.path.join(OUT_DIR, 'figure2_sentiment_timeseries.pdf')


FINBERT_COLOR = '#1a1a1a'
LM_COLOR = '#888888'
CUT_COLOR = '#b04a4a'
HOLD_COLOR = '#999999'
HIKE_COLOR = '#4f7f4f'
RATE_LINE_COLOR = '#444444'


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
        'legend.fontsize': 9.5,
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


def load_data():
    """Load and validate the final meeting-level dataset."""
    df = pd.read_csv(MASTER_CSV)
    df['meeting_date'] = pd.to_datetime(df['meeting_date'])
    df = df.sort_values('meeting_date').reset_index(drop=True)

    required_cols = [
        'meeting_date',
        'lm_sentiment_std',
        'finbert_sentiment_std',
        'rate_change_bps',
        'lagged_rate',
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}')

    # Reconstruct the policy rate after each meeting decision.
    df['policy_rate'] = df['lagged_rate'] + df['rate_change_bps'] / 100

    return df


def format_axes(ax):
    """Apply consistent axis formatting."""
    ax.grid(True, axis='y', linewidth=0.6)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    ax.tick_params(colors='#333333')


def build_figure(df):
    """Create the two-panel figure."""
    fig, (ax_sent, ax_rate) = plt.subplots(
        2,
        1,
        figsize=(11, 7.0),
        sharex=True,
        gridspec_kw={'height_ratios': [1.6, 1], 'hspace': 0.18},
    )

    ax_sent.plot(
        df['meeting_date'],
        df['finbert_sentiment_std'],
        color=FINBERT_COLOR,
        linewidth=1.8,
        label='FinBERT sentiment (std.)',
        zorder=3,
    )
    ax_sent.plot(
        df['meeting_date'],
        df['lm_sentiment_std'],
        color=LM_COLOR,
        linewidth=1.5,
        linestyle='--',
        label='LM sentiment (std.)',
        zorder=2,
    )

    # Rolling averages show the medium-term trend.
    window = 12
    fb_ma = df['finbert_sentiment_std'].rolling(window, center=True, min_periods=4).mean()
    lm_ma = df['lm_sentiment_std'].rolling(window, center=True, min_periods=4).mean()

    ax_sent.plot(
        df['meeting_date'], fb_ma,
        color='#c44e52', linewidth=1.4, zorder=4,
        label=f'FinBERT trend ({window}-meeting avg.)',
    )
    ax_sent.plot(
        df['meeting_date'], lm_ma,
        color='#4c72b0', linewidth=1.4, zorder=4,
        label=f'LM trend ({window}-meeting avg.)',
    )

    ax_sent.axhline(0, color='#b0b0b0', linewidth=0.8, linestyle=':')
    ax_sent.set_ylabel('Standardized sentiment')
    ax_sent.set_title('Panel A: Meeting-level sentiment measures', loc='left',
                      fontweight='bold', pad=8)
    ax_sent.legend(frameon=False, loc='upper left', ncol=2, fontsize=9)
    format_axes(ax_sent)

    ax_rate.plot(
        df['meeting_date'],
        df['policy_rate'],
        color=RATE_LINE_COLOR,
        linewidth=1.0,
        zorder=1,
    )

    cuts = df[df['rate_change_bps'] < 0]
    holds = df[df['rate_change_bps'] == 0]
    hikes = df[df['rate_change_bps'] > 0]

    ax_rate.scatter(
        cuts['meeting_date'], cuts['policy_rate'],
        s=40, color=CUT_COLOR, edgecolor='white', linewidth=0.5,
        zorder=3, label='Cut',
    )
    ax_rate.scatter(
        holds['meeting_date'], holds['policy_rate'],
        s=22, color=HOLD_COLOR, edgecolor='white', linewidth=0.4,
        zorder=2, label='Hold',
    )
    ax_rate.scatter(
        hikes['meeting_date'], hikes['policy_rate'],
        s=40, color=HIKE_COLOR, edgecolor='white', linewidth=0.5,
        zorder=3, label='Hike',
    )

    ax_rate.set_ylabel('Policy rate (%)')
    ax_rate.set_title('Panel B: Norges Bank policy rate', loc='left',
                      fontweight='bold', pad=8)
    ax_rate.set_ylim(-0.3, df['policy_rate'].max() + 0.5)
    format_axes(ax_rate)

    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=CUT_COLOR,
               markeredgecolor='white', markeredgewidth=0.5, markersize=6.5, label='Cut'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=HOLD_COLOR,
               markeredgecolor='white', markeredgewidth=0.4, markersize=5, label='Hold'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=HIKE_COLOR,
               markeredgecolor='white', markeredgewidth=0.5, markersize=6.5, label='Hike'),
    ]
    ax_rate.legend(handles=legend_handles, frameon=False, loc='upper right', ncol=3)

    ax_rate.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_rate.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax_rate.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_rate.set_xlabel('')
    fig.autofmt_xdate(rotation=0, ha='center')

    fig.align_ylabels([ax_sent, ax_rate])

    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_plot_style()

    df = load_data()

    fig = build_figure(df)
    fig.savefig(PNG_OUT, dpi=600)
    fig.savefig(PDF_OUT)
    plt.close(fig)

    print(f'PNG: {PNG_OUT}')
    print(f'PDF: {PDF_OUT}')


if __name__ == '__main__':
    main()