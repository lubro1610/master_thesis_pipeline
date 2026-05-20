"""Pre-2020 robustness check for the ordered probit models."""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('02_Data/master_dataset_with_sentiment.csv')
df['meeting_date'] = pd.to_datetime(df['meeting_date'])
df['year'] = df['meeting_date'].dt.year

key_vars = ['decision_ordinal', 'inflation_gap', 'output_next', 'lagged_rate',
            'lm_sentiment_std', 'finbert_sentiment_std']
df_clean = df[key_vars + ['year', 'meeting_date']].dropna()

df_full = df_clean.copy()
df_pre2020 = df_clean[df_clean['year'] < 2020].copy()

y_full = df_full['decision_ordinal']

# Full sample models
X1_full = df_full[['inflation_gap', 'output_next', 'lagged_rate']]
model1_full = OrderedModel(y_full, X1_full, distr='probit')
result1_full = model1_full.fit(method='bfgs', disp=False)

X2_full = df_full[['inflation_gap', 'output_next', 'lagged_rate', 'lm_sentiment_std']]
model2_full = OrderedModel(y_full, X2_full, distr='probit')
result2_full = model2_full.fit(method='bfgs', disp=False)

X3_full = df_full[['inflation_gap', 'output_next', 'lagged_rate', 'finbert_sentiment_std']]
model3_full = OrderedModel(y_full, X3_full, distr='probit')
result3_full = model3_full.fit(method='bfgs', disp=False)

X4_full = df_full[['inflation_gap', 'output_next', 'lagged_rate',
                   'lm_sentiment_std', 'finbert_sentiment_std']]
model4_full = OrderedModel(y_full, X4_full, distr='probit')
result4_full = model4_full.fit(method='bfgs', disp=False)

y_pre2020 = df_pre2020['decision_ordinal']

# Pre-2020 models
X1_pre2020 = df_pre2020[['inflation_gap', 'output_next', 'lagged_rate']]
model1_pre2020 = OrderedModel(y_pre2020, X1_pre2020, distr='probit')
result1_pre2020 = model1_pre2020.fit(method='bfgs', disp=False)

X2_pre2020 = df_pre2020[['inflation_gap', 'output_next', 'lagged_rate', 'lm_sentiment_std']]
model2_pre2020 = OrderedModel(y_pre2020, X2_pre2020, distr='probit')
result2_pre2020 = model2_pre2020.fit(method='bfgs', disp=False)

X3_pre2020 = df_pre2020[['inflation_gap', 'output_next', 'lagged_rate', 'finbert_sentiment_std']]
model3_pre2020 = OrderedModel(y_pre2020, X3_pre2020, distr='probit')
result3_pre2020 = model3_pre2020.fit(method='bfgs', disp=False)

X4_pre2020 = df_pre2020[['inflation_gap', 'output_next', 'lagged_rate',
                         'lm_sentiment_std', 'finbert_sentiment_std']]
model4_pre2020 = OrderedModel(y_pre2020, X4_pre2020, distr='probit')
result4_pre2020 = model4_pre2020.fit(method='bfgs', disp=False)

lm_full_coef = result2_full.params['lm_sentiment_std']
lm_full_pval = result2_full.pvalues['lm_sentiment_std']
lm_full_sig = "***" if lm_full_pval < 0.01 else "**" if lm_full_pval < 0.05 else "*" if lm_full_pval < 0.10 else ""

lm_pre_coef = result2_pre2020.params['lm_sentiment_std']
lm_pre_pval = result2_pre2020.pvalues['lm_sentiment_std']
lm_pre_sig = "***" if lm_pre_pval < 0.01 else "**" if lm_pre_pval < 0.05 else "*" if lm_pre_pval < 0.10 else ""

change_lm = ((lm_pre_coef - lm_full_coef) / abs(lm_full_coef) * 100) if lm_full_coef != 0 else 0

fb_full_coef = result3_full.params['finbert_sentiment_std']
fb_full_pval = result3_full.pvalues['finbert_sentiment_std']
fb_full_sig = "***" if fb_full_pval < 0.01 else "**" if fb_full_pval < 0.05 else "*" if fb_full_pval < 0.10 else ""

fb_pre_coef = result3_pre2020.params['finbert_sentiment_std']
fb_pre_pval = result3_pre2020.pvalues['finbert_sentiment_std']
fb_pre_sig = "***" if fb_pre_pval < 0.01 else "**" if fb_pre_pval < 0.05 else "*" if fb_pre_pval < 0.10 else ""

change_fb = ((fb_pre_coef - fb_full_coef) / abs(fb_full_coef) * 100) if fb_full_coef != 0 else 0

lm_full_comb = result4_full.params['lm_sentiment_std']
lm_full_comb_pval = result4_full.pvalues['lm_sentiment_std']
fb_full_comb = result4_full.params['finbert_sentiment_std']
fb_full_comb_pval = result4_full.pvalues['finbert_sentiment_std']

lm_pre_comb = result4_pre2020.params['lm_sentiment_std']
lm_pre_comb_pval = result4_pre2020.pvalues['lm_sentiment_std']
fb_pre_comb = result4_pre2020.params['finbert_sentiment_std']
fb_pre_comb_pval = result4_pre2020.pvalues['finbert_sentiment_std']

models_full = {
    'Baseline': result1_full,
    'LM Only': result2_full,
    'FinBERT Only': result3_full,
    'Combined': result4_full
}

models_pre2020 = {
    'Baseline': result1_pre2020,
    'LM Only': result2_pre2020,
    'FinBERT Only': result3_pre2020,
    'Combined': result4_pre2020
}

lr_lm_full = 2 * (result2_full.llf - result1_full.llf)
lr_lm_full_pval = stats.chi2.sf(lr_lm_full, 1)

lr_fb_full = 2 * (result3_full.llf - result1_full.llf)
lr_fb_full_pval = stats.chi2.sf(lr_fb_full, 1)

lr_lm_pre = 2 * (result2_pre2020.llf - result1_pre2020.llf)
lr_lm_pre_pval = stats.chi2.sf(lr_lm_pre, 1)

lr_fb_pre = 2 * (result3_pre2020.llf - result1_pre2020.llf)
lr_fb_pre_pval = stats.chi2.sf(lr_fb_pre, 1)


def compute_marginal_effects(result, X, var_name):
    """Compute average marginal effects for an ordered probit model."""
    params = result.params
    threshold_0, threshold_1 = result.model.transform_threshold_params(params.values)[1:-1]
    beta = params.drop(['-1/0', '0/1'])

    xb = X.values @ beta.values
    phi_0 = norm.pdf(threshold_0 - xb)
    phi_1 = norm.pdf(threshold_1 - xb)
    Phi_0 = norm.cdf(threshold_0 - xb)
    Phi_1 = norm.cdf(threshold_1 - xb)

    beta_k = beta[var_name]

    me_cut = (-phi_0 * beta_k).mean()
    me_hold = ((phi_0 - phi_1) * beta_k).mean()
    me_hike = (phi_1 * beta_k).mean()

    prob_cut = Phi_0.mean()
    prob_hold = (Phi_1 - Phi_0).mean()
    prob_hike = (1 - Phi_1).mean()

    return {
        'me_cut': me_cut,
        'me_hold': me_hold,
        'me_hike': me_hike,
        'prob_cut': prob_cut,
        'prob_hold': prob_hold,
        'prob_hike': prob_hike
    }


me_fb_full = compute_marginal_effects(result3_full, X3_full, 'finbert_sentiment_std')
me_fb_pre = compute_marginal_effects(result3_pre2020, X3_pre2020, 'finbert_sentiment_std')
me_change_hike = ((me_fb_pre['me_hike'] - me_fb_full['me_hike']) / abs(me_fb_full['me_hike']) * 100)

me_lm_full = compute_marginal_effects(result2_full, X2_full, 'lm_sentiment_std')
me_lm_pre = compute_marginal_effects(result2_pre2020, X2_pre2020, 'lm_sentiment_std')

me_fb_full_comb = compute_marginal_effects(result4_full, X4_full, 'finbert_sentiment_std')
me_lm_full_comb = compute_marginal_effects(result4_full, X4_full, 'lm_sentiment_std')

me_fb_pre_comb = compute_marginal_effects(result4_pre2020, X4_pre2020, 'finbert_sentiment_std')
me_lm_pre_comb = compute_marginal_effects(result4_pre2020, X4_pre2020, 'lm_sentiment_std')

results_data = []

results_data.append({
    'Sample': 'Full (2006-2026)',
    'N': len(df_full),
    'LM_coef': lm_full_coef,
    'LM_pval': lm_full_pval,
    'LM_sig': lm_full_sig,
    'FB_coef': fb_full_coef,
    'FB_pval': fb_full_pval,
    'FB_sig': fb_full_sig,
    'LM_comb': lm_full_comb,
    'LM_comb_pval': lm_full_comb_pval,
    'FB_comb': fb_full_comb,
    'FB_comb_pval': fb_full_comb_pval,
    'FB_ME_Hike_pp': me_fb_full['me_hike'] * 100,
    'LM_ME_Hike_pp': me_lm_full['me_hike'] * 100
})

results_data.append({
    'Sample': 'Pre-2020 (2006-2019)',
    'N': len(df_pre2020),
    'LM_coef': lm_pre_coef,
    'LM_pval': lm_pre_pval,
    'LM_sig': lm_pre_sig,
    'FB_coef': fb_pre_coef,
    'FB_pval': fb_pre_pval,
    'FB_sig': fb_pre_sig,
    'LM_comb': lm_pre_comb,
    'LM_comb_pval': lm_pre_comb_pval,
    'FB_comb': fb_pre_comb,
    'FB_comb_pval': fb_pre_comb_pval,
    'FB_ME_Hike_pp': me_fb_pre['me_hike'] * 100,
    'LM_ME_Hike_pp': me_lm_pre['me_hike'] * 100
})

results_df = pd.DataFrame(results_data)
results_df.to_csv('04_Output/robustness_pre2020.csv', index=False)

ame_data = [
    {
        'Sample': 'Full (2006-2026)',
        'Model': 'LM Only',
        'Variable': 'lm_sentiment_std',
        'Coefficient': lm_full_coef,
        'p_value': lm_full_pval,
        'ME_Cut_pp': me_lm_full['me_cut'] * 100,
        'ME_Hold_pp': me_lm_full['me_hold'] * 100,
        'ME_Hike_pp': me_lm_full['me_hike'] * 100,
        'Prob_Cut': me_lm_full['prob_cut'] * 100,
        'Prob_Hold': me_lm_full['prob_hold'] * 100,
        'Prob_Hike': me_lm_full['prob_hike'] * 100
    },
    {
        'Sample': 'Full (2006-2026)',
        'Model': 'FinBERT Only',
        'Variable': 'finbert_sentiment_std',
        'Coefficient': fb_full_coef,
        'p_value': fb_full_pval,
        'ME_Cut_pp': me_fb_full['me_cut'] * 100,
        'ME_Hold_pp': me_fb_full['me_hold'] * 100,
        'ME_Hike_pp': me_fb_full['me_hike'] * 100,
        'Prob_Cut': me_fb_full['prob_cut'] * 100,
        'Prob_Hold': me_fb_full['prob_hold'] * 100,
        'Prob_Hike': me_fb_full['prob_hike'] * 100
    },
    {
        'Sample': 'Full (2006-2026)',
        'Model': 'Combined',
        'Variable': 'lm_sentiment_std',
        'Coefficient': lm_full_comb,
        'p_value': lm_full_comb_pval,
        'ME_Cut_pp': me_lm_full_comb['me_cut'] * 100,
        'ME_Hold_pp': me_lm_full_comb['me_hold'] * 100,
        'ME_Hike_pp': me_lm_full_comb['me_hike'] * 100,
        'Prob_Cut': me_lm_full_comb['prob_cut'] * 100,
        'Prob_Hold': me_lm_full_comb['prob_hold'] * 100,
        'Prob_Hike': me_lm_full_comb['prob_hike'] * 100
    },
    {
        'Sample': 'Full (2006-2026)',
        'Model': 'Combined',
        'Variable': 'finbert_sentiment_std',
        'Coefficient': fb_full_comb,
        'p_value': fb_full_comb_pval,
        'ME_Cut_pp': me_fb_full_comb['me_cut'] * 100,
        'ME_Hold_pp': me_fb_full_comb['me_hold'] * 100,
        'ME_Hike_pp': me_fb_full_comb['me_hike'] * 100,
        'Prob_Cut': me_fb_full_comb['prob_cut'] * 100,
        'Prob_Hold': me_fb_full_comb['prob_hold'] * 100,
        'Prob_Hike': me_fb_full_comb['prob_hike'] * 100
    },
    {
        'Sample': 'Pre-2020 (2006-2019)',
        'Model': 'LM Only',
        'Variable': 'lm_sentiment_std',
        'Coefficient': lm_pre_coef,
        'p_value': lm_pre_pval,
        'ME_Cut_pp': me_lm_pre['me_cut'] * 100,
        'ME_Hold_pp': me_lm_pre['me_hold'] * 100,
        'ME_Hike_pp': me_lm_pre['me_hike'] * 100,
        'Prob_Cut': me_lm_pre['prob_cut'] * 100,
        'Prob_Hold': me_lm_pre['prob_hold'] * 100,
        'Prob_Hike': me_lm_pre['prob_hike'] * 100
    },
    {
        'Sample': 'Pre-2020 (2006-2019)',
        'Model': 'FinBERT Only',
        'Variable': 'finbert_sentiment_std',
        'Coefficient': fb_pre_coef,
        'p_value': fb_pre_pval,
        'ME_Cut_pp': me_fb_pre['me_cut'] * 100,
        'ME_Hold_pp': me_fb_pre['me_hold'] * 100,
        'ME_Hike_pp': me_fb_pre['me_hike'] * 100,
        'Prob_Cut': me_fb_pre['prob_cut'] * 100,
        'Prob_Hold': me_fb_pre['prob_hold'] * 100,
        'Prob_Hike': me_fb_pre['prob_hike'] * 100
    },
    {
        'Sample': 'Pre-2020 (2006-2019)',
        'Model': 'Combined',
        'Variable': 'lm_sentiment_std',
        'Coefficient': lm_pre_comb,
        'p_value': lm_pre_comb_pval,
        'ME_Cut_pp': me_lm_pre_comb['me_cut'] * 100,
        'ME_Hold_pp': me_lm_pre_comb['me_hold'] * 100,
        'ME_Hike_pp': me_lm_pre_comb['me_hike'] * 100,
        'Prob_Cut': me_lm_pre_comb['prob_cut'] * 100,
        'Prob_Hold': me_lm_pre_comb['prob_hold'] * 100,
        'Prob_Hike': me_lm_pre_comb['prob_hike'] * 100
    },
    {
        'Sample': 'Pre-2020 (2006-2019)',
        'Model': 'Combined',
        'Variable': 'finbert_sentiment_std',
        'Coefficient': fb_pre_comb,
        'p_value': fb_pre_comb_pval,
        'ME_Cut_pp': me_fb_pre_comb['me_cut'] * 100,
        'ME_Hold_pp': me_fb_pre_comb['me_hold'] * 100,
        'ME_Hike_pp': me_fb_pre_comb['me_hike'] * 100,
        'Prob_Cut': me_fb_pre_comb['prob_cut'] * 100,
        'Prob_Hold': me_fb_pre_comb['prob_hold'] * 100,
        'Prob_Hike': me_fb_pre_comb['prob_hike'] * 100
    }
]

ame_table = pd.DataFrame(ame_data)

print("Pre-2020 robustness AMEs, percentage points:")
print(ame_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("Saved: 04_Output/robustness_pre2020.csv")
