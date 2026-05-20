"""Generate Figure 4: document-category sentiment coefficients."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))

COEF_CSV = os.path.join(WORKSPACE, '04_Output', 'robustness_document_types.csv')
OUT_DIR = os.path.join(WORKSPACE, '04_Output', 'figures')

PNG_OUT = os.path.join(OUT_DIR, 'figure4_document_categories.png')
PDF_OUT = os.path.join(OUT_DIR, 'figure4_document_categories.pdf')


FINBERT_COLOR = '#2c2c2c'
LM_COLOR = '#a0a0a0'

LABEL_MAP = {
    'mpr': 'Monetary Policy\nReport',
    'banklend': 'Bank Lending\nSurvey',
    'speeches': 'Speeches',
    'press_releases': 'Press Releases',
}

# Display the most informative categories first.
CATEGORY_ORDER = ['mpr', 'banklend', 'speeches', 'press_releases']


def sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return ''


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


def load_data():
    """Load coefficient data and pivot into FinBERT / LM columns."""
    df = pd.read_csv(COEF_CSV)

    fb = df[df['method'] == 'FinBERT'].set_index('doc_type')
    lm = df[df['method'] == 'LM'].set_index('doc_type')

    rows = []
    for cat in CATEGORY_ORDER:
        rows.append({
            'doc_type': cat,
            'label': LABEL_MAP[cat],
            'fb_coef': fb.loc[cat, 'coef'],
            'fb_pval': fb.loc[cat, 'pval'],
            'lm_coef': lm.loc[cat, 'coef'],
            'lm_pval': lm.loc[cat, 'pval'],
            'n': int(fb.loc[cat, 'n']),
        })

    return pd.DataFrame(rows)


def build_figure(data):
    """Create the grouped bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(data))
    bar_width = 0.32

    bars_fb = ax.bar(
        x - bar_width / 2, data['fb_coef'], bar_width,
        color=FINBERT_COLOR, label='FinBERT', zorder=3,
    )
    bars_lm = ax.bar(
        x + bar_width / 2, data['lm_coef'], bar_width,
        color=LM_COLOR, label='LM', zorder=3,
    )

    for i, row in data.iterrows():
        stars_fb = sig_stars(row['fb_pval'])
        label_fb = f'{row["fb_coef"]:.3f}{stars_fb}'
        ax.text(
            x[i] - bar_width / 2, row['fb_coef'] + 0.02,
            label_fb, ha='center', va='bottom', fontsize=8,
        )
        stars_lm = sig_stars(row['lm_pval'])
        label_lm = f'{row["lm_coef"]:.3f}{stars_lm}'
        ax.text(
            x[i] + bar_width / 2, row['lm_coef'] + 0.02,
            label_lm, ha='center', va='bottom', fontsize=8,
        )

    ax.set_xticks(x)
    labels = [f'{row["label"]}\n($\\it{{N}}$ = {row["n"]})' for _, row in data.iterrows()]
    ax.set_xticklabels(labels)
    ax.set_ylabel('Ordered probit coefficient')
    ax.axhline(0, color='#b0b0b0', linewidth=0.8, linestyle='-')

    ax.grid(True, axis='y', linewidth=0.6)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    ax.tick_params(colors='#333333')

    ax.legend(frameon=False, loc='upper right')

    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_plot_style()

    data = load_data()
    for _, row in data.iterrows():
        fb_s = sig_stars(row['fb_pval'])
        lm_s = sig_stars(row['lm_pval'])
        print(f'{row["doc_type"]:<16} FinBERT={row["fb_coef"]:.3f}{fb_s:<4} '
              f'LM={row["lm_coef"]:.3f}{lm_s:<4}  (n={row["n"]})')

    fig = build_figure(data)
    fig.savefig(PNG_OUT, dpi=600)
    fig.savefig(PDF_OUT)
    plt.close(fig)

    print(f'PNG: {PNG_OUT}')
    print(f'PDF: {PDF_OUT}')


if __name__ == '__main__':
    main()
