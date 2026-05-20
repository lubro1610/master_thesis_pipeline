"""Lagged-decision robustness check for Models 2 and 3."""

import os
import sys
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE  = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))
MASTER     = os.path.join(WORKSPACE, '02_Data', 'master_dataset_with_sentiment.csv')

df = pd.read_csv(MASTER)
df['meeting_date'] = pd.to_datetime(df['meeting_date'])
df = df.sort_values('meeting_date').reset_index(drop=True)

# Previous meeting decision as an ordinal smoothing variable.
df['lagged_decision'] = df['decision_ordinal'].shift(1)

# The first in-sample meeting needs the previous out-of-sample decision.
first_idx = df.sort_values('meeting_date').index[0]
df.loc[first_idx, 'lagged_decision'] = 0

key_vars = ['decision_ordinal', 'inflation_gap', 'output_next',
            'lagged_decision',
            'lm_sentiment_std', 'finbert_sentiment_std']
df = df.dropna(subset=key_vars).copy()
df['lagged_decision'] = df['lagged_decision'].astype(int)

print(f"n={len(df)}")
print(f"period={df['meeting_date'].min().date()} to {df['meeting_date'].max().date()}")
print("lagged_decision distribution:")
print(df['lagged_decision'].value_counts().sort_index())

y = df['decision_ordinal']


def get_actual_cutpoints(result):
    cutpoints = result.model.transform_threshold_params(result.params.values)[1:-1]
    return cutpoints[0], cutpoints[1]


def print_actual_cutpoints(result):
    tau_1, tau_2 = get_actual_cutpoints(result)
    print(f"cutpoints: tau_1={tau_1:.6f}, tau_2={tau_2:.6f}")

# Null model has only cutpoints.
counts  = y.value_counts()
ll_null = float(sum(counts[c] * np.log(counts[c] / len(y)) for c in counts.index))

models = {
    'Model 2: LM + lagged_decision': [
        'inflation_gap', 'output_next', 'lagged_decision', 'lm_sentiment_std'
    ],
    'Model 3: FinBERT + lagged_decision': [
        'inflation_gap', 'output_next', 'lagged_decision', 'finbert_sentiment_std'
    ],
}

for model_name, columns in models.items():
    print(f"\n{model_name}")

    X = df[columns]
    res = OrderedModel(y, X, distr='probit').fit(method='bfgs', disp=False)
    print(res.summary())
    print_actual_cutpoints(res)

    mcfadden = 1 - res.llf / ll_null
    print(f"McFadden pseudo-R2: {mcfadden:.4f}")
