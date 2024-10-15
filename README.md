# Scope and Limitations of Market Generators

Overview of the code used in the Master's thesis. All plots and evaluations in the thesis can be replicated using this repository.

## Used Containers for Server Computing:
- Liao Code: nvcr.io#nvidia/pytorch:20.01-py3
- Buehler Code: nvcr.io#nvidia/tensorflow:20.03-tf1-py3

## Structure
We forked the repositories from Buehler et al. (CVAE) and Liao et al. (SigCWGAN and other GANs) into this repo and added our own evaluations and inputs to their logic. All code can be run on Jupyter Notebooks in the "notebooks" folder. The code is written to work on the LRZ server but can be easily run locally, too. Just specify the correct working directory at the beginning of the respective script.

### Forked Repos
- "BuehlerVAE/"
- "LiaoWGAN/"

### Own Code
- "notebooks/"
  - "Buehler/"
    - Train the model and generate (signature-level) data with `market_generator_synthetic_data.ipynb` and `market_generator_YFinance.ipynb` for different types of input.
    - Invert the generated signatures from the previous scripts with `market_generator_inversion_synthetic.ipynb` and `market_generator_inversion_server_YFinance.ipynb` and save them in the `numerical_results` folder structure of Liao to ensure unified comparison.
    - Test the code (locally) with `market_generator_intro_gbm.ipynb`.
  - "Liao/"
    - Run all models evaluated in the paper from Liao et al. (including SigCWGAN) using the `RunLiao.ipynb` script. Make sure to specify the correct GPU or CPU, depending on available resources. This file creates the necessary file structure dynamically itself.
    - Get evaluation metrics from Liao et al. paper summarized in CSVs via `evaluateLiao.ipynb`.
  - "helper/"
    - Various helper functionalities to create histograms, sort files in folders, and code for the visualization of Section 2.1 of the thesis.
  - "option_pricing/"
    - This folder contains the logic for pricing the relevant options and their respective plots based on the generated data in the `numerical_results` folder structure. Since the structure of the training data differs, we distinguish between data from Yahoo Finance and synthetic data created by ourselves.
    - `OneTimeTrained_Synthetic.ipynb` and `OneTimeTrained_YFinance.ipynb` create the one-time trained models' plots listed in the thesis. This script may run for multiple hours, depending on the output sizes for the MC simulations.
    - The calculations for the variational analysis for the retrained models are performed via `StatAnalysis_ModelRetrained_Synthetic.ipynb` and `StatAnalysis_ModelRetrained_YFinance.ipynb` and the results are saved to a CSV file.
    - The calculations for the variational analysis for the one-time trained models are performed via `StatAnalysis_SameModel_Synthetic.ipynb` and `StatAnalysis_SameModel_YFinance.ipynb` and the results are saved to a CSV file.
    - The plots for the variational analysis based on the created CSVs can be created with the `PlotStatAnalysis.ipynb` file.

- "src/" contains all logic that is called in the notebooks. See respective files for details.

    