"""Average marginal effects for the ordered probit models."""

import sys
import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('02_Data/master_dataset_with_sentiment.csv')
df['meeting_date'] = pd.to_datetime(df['meeting_date'])

key_vars = ['decision_ordinal', 'inflation_gap', 'output_next', 'lagged_rate', 
            'lm_sentiment_std', 'finbert_sentiment_std']
df_clean = df[key_vars].dropna()
y = df_clean['decision_ordinal']

# Model 1: Baseline
X1 = df_clean[['inflation_gap', 'output_next', 'lagged_rate']]
model1 = OrderedModel(y, X1, distr='probit')
result1 = model1.fit(method='bfgs', disp=False)

# Model 2: LM Sentiment
X2 = df_clean[['inflation_gap', 'output_next', 'lagged_rate', 'lm_sentiment_std']]
model2 = OrderedModel(y, X2, distr='probit')
result2 = model2.fit(method='bfgs', disp=False)

# Model 3: FinBERT Sentiment
X3 = df_clean[['inflation_gap', 'output_next', 'lagged_rate', 'finbert_sentiment_std']]
model3 = OrderedModel(y, X3, distr='probit')
result3 = model3.fit(method='bfgs', disp=False)

# Model 4: Combined
X4 = df_clean[['inflation_gap', 'output_next', 'lagged_rate', 
               'lm_sentiment_std', 'finbert_sentiment_std']]
model4 = OrderedModel(y, X4, distr='probit')
result4 = model4.fit(method='bfgs', disp=False)

def compute_marginal_effects(result, X, var_name):
    """Compute average marginal effects for an ordered probit model."""
    # OrderedModel stores thresholds after the first as transformed increments.
    params = result.params
    threshold_0, threshold_1 = result.model.transform_threshold_params(params.values)[1:-1]
    
    beta = params.drop(['-1/0', '0/1'])
    xb = X.values @ beta.values
    phi_0 = norm.pdf(threshold_0 - xb)
    phi_1 = norm.pdf(threshold_1 - xb)
    Phi_0 = norm.cdf(threshold_0 - xb)
    Phi_1 = norm.cdf(threshold_1 - xb)

    beta_k = beta[var_name]
    me_cut_i = -phi_0 * beta_k
    me_hold_i = (phi_0 - phi_1) * beta_k
    me_hike_i = phi_1 * beta_k

    me_cut = me_cut_i.mean()
    me_hold = me_hold_i.mean()
    me_hike = me_hike_i.mean()

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

# Model 2: LM Sentiment
me_lm = compute_marginal_effects(result2, X2, 'lm_sentiment_std')
lm_coef = result2.params['lm_sentiment_std']
lm_pval = result2.pvalues['lm_sentiment_std']
lm_sig = "***" if lm_pval < 0.01 else "**" if lm_pval < 0.05 else "*" if lm_pval < 0.10 else ""

# Model 3: FinBERT Sentiment
me_fb = compute_marginal_effects(result3, X3, 'finbert_sentiment_std')
fb_coef = result3.params['finbert_sentiment_std']
fb_pval = result3.pvalues['finbert_sentiment_std']
fb_sig = "***" if fb_pval < 0.01 else "**" if fb_pval < 0.05 else "*" if fb_pval < 0.10 else ""

# Model 4: Combined
me_lm_comb = compute_marginal_effects(result4, X4, 'lm_sentiment_std')
me_fb_comb = compute_marginal_effects(result4, X4, 'finbert_sentiment_std')
lm_comb_coef = result4.params['lm_sentiment_std']
lm_comb_pval = result4.pvalues['lm_sentiment_std']
lm_comb_sig = "***" if lm_comb_pval < 0.01 else "**" if lm_comb_pval < 0.05 else "*" if lm_comb_pval < 0.10 else ""
fb_comb_coef = result4.params['finbert_sentiment_std']
fb_comb_pval = result4.pvalues['finbert_sentiment_std']
fb_comb_sig = "***" if fb_comb_pval < 0.01 else "**" if fb_comb_pval < 0.05 else "*" if fb_comb_pval < 0.10 else ""

# Create marginal effects table
me_data = []

# Model 2: LM
me_data.append({
    'Model': 'LM Only',
    'Variable': 'lm_sentiment_std',
    'Coefficient': result2.params['lm_sentiment_std'],
    'p_value': result2.pvalues['lm_sentiment_std'],
    'ME_Cut_pp': me_lm['me_cut'] * 100,
    'ME_Hold_pp': me_lm['me_hold'] * 100,
    'ME_Hike_pp': me_lm['me_hike'] * 100,
    'Prob_Cut': me_lm['prob_cut'] * 100,
    'Prob_Hold': me_lm['prob_hold'] * 100,
    'Prob_Hike': me_lm['prob_hike'] * 100
})

# Model 3: FinBERT
me_data.append({
    'Model': 'FinBERT Only',
    'Variable': 'finbert_sentiment_std',
    'Coefficient': result3.params['finbert_sentiment_std'],
    'p_value': result3.pvalues['finbert_sentiment_std'],
    'ME_Cut_pp': me_fb['me_cut'] * 100,
    'ME_Hold_pp': me_fb['me_hold'] * 100,
    'ME_Hike_pp': me_fb['me_hike'] * 100,
    'Prob_Cut': me_fb['prob_cut'] * 100,
    'Prob_Hold': me_fb['prob_hold'] * 100,
    'Prob_Hike': me_fb['prob_hike'] * 100
})

# Model 4: Combined - LM
me_data.append({
    'Model': 'Combined',
    'Variable': 'lm_sentiment_std',
    'Coefficient': result4.params['lm_sentiment_std'],
    'p_value': result4.pvalues['lm_sentiment_std'],
    'ME_Cut_pp': me_lm_comb['me_cut'] * 100,
    'ME_Hold_pp': me_lm_comb['me_hold'] * 100,
    'ME_Hike_pp': me_lm_comb['me_hike'] * 100,
    'Prob_Cut': me_lm_comb['prob_cut'] * 100,
    'Prob_Hold': me_lm_comb['prob_hold'] * 100,
    'Prob_Hike': me_lm_comb['prob_hike'] * 100
})

# Model 4: Combined - FinBERT
me_data.append({
    'Model': 'Combined',
    'Variable': 'finbert_sentiment_std',
    'Coefficient': result4.params['finbert_sentiment_std'],
    'p_value': result4.pvalues['finbert_sentiment_std'],
    'ME_Cut_pp': me_fb_comb['me_cut'] * 100,
    'ME_Hold_pp': me_fb_comb['me_hold'] * 100,
    'ME_Hike_pp': me_fb_comb['me_hike'] * 100,
    'Prob_Cut': me_fb_comb['prob_cut'] * 100,
    'Prob_Hold': me_fb_comb['prob_hold'] * 100,
    'Prob_Hike': me_fb_comb['prob_hike'] * 100
})

me_table = pd.DataFrame(me_data)
me_table.to_csv('04_Output/marginal_effects.csv', index=False)

print("Average marginal effects, percentage points:")
print(me_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("Saved: 04_Output/marginal_effects.csv")
