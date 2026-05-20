# Master thesis code and replication files

This repository contains the code and selected replication files for my master thesis analysis of Norges Bank communication sentiment and monetary policy decisions.

## Contents

- `03_Code/`: analysis pipeline code.
- `02_Data/master_dataset_with_sentiment.csv`: final meeting-level dataset with macro variables and sentiment measures. This is the most convenient starting point: using this file, the main regressions and figures can be run without re-running the full harvesting, extraction, filtering, and aggregation pipeline.
- `Environment/`: supporting CSV files used by the pipeline, including control variables, dictionary inputs, and meeting-level macro data.
- `04_Output/`: empty output folders for generated regression outputs and figures.

## Reproducing the main analysis

The fastest path is to use the included final dataset:

```powershell
python 03_Code/08_Regression/01_ordered_probit_regression.py
python 03_Code/08_Regression/03_marginal_effects.py
python 03_Code/09_Figures/07_generate_figure3_predicted_probabilities.py
```

The full pipeline code is included, but the repository does not include all raw harvested text files or large intermediate corpora. I have included the final master dataset instead, since that is much more convenient for replication. The harvesting and extraction stages can be re-run if needed, but they depend on current website structure and network availability.

## Environment

Install the Python dependencies with:

```powershell
pip install -r requirements.txt
```

The code was run with Python 3.13.7. The package versions used in my submitted environment are pinned in `requirements.txt`. The requirements file includes the PyTorch CUDA wheel index used for the GPU environment.

## Runtime and Hardware Notes

- The harvesting, extraction, sentence-level corpora, and sentiment-scoring stages can take a long time and generate many intermediate CSV files. I therefore treat the included final dataset as the practical replication entry point.
- FinBERT sentence-level sentiment scoring uses Hugging Face Transformers and PyTorch. I ran it with CUDA-enabled PyTorch (`torch==2.10.0+cu128`) on a local NVIDIA GPU, which made a very large difference for inference time. The script falls back to CPU if no GPU is available, but CPU execution is expected to be very slow for the full sentence corpus.
- The included final dataset is therefore the recommended starting point for reproducing the thesis regressions and figures.

## Software Versions

The submitted environment used:

- Python 3.13.7
- pandas 2.3.3
- NumPy 2.4.1
- SciPy 1.17.0
- statsmodels 0.14.6
- Matplotlib 3.10.8
- Requests 2.32.5
- Beautiful Soup 4.14.3
- PyMuPDF 1.26.7
- tqdm 4.67.1
- Selenium 4.40.0
- webdriver-manager 4.0.2
- NLTK 3.9.2
- Transformers 5.1.0
- PyTorch 2.10.0+cu128

## Additional Reproducibility Notes

- The main empirical scripts read from `02_Data/master_dataset_with_sentiment.csv` and write outputs to `04_Output/`.
- The full pipeline is organized by numbered folders in `03_Code/`, from harvesting through figures. The numbering reflects the order in which the thesis pipeline developed, and the scripts assume they are run from the repository root.
- Some upstream collection scripts are inherently time-sensitive because public web pages can change after thesis submission. At the time of writing, the scripts reflected the website structure used during the thesis work.


