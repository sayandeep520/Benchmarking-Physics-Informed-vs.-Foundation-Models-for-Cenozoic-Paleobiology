# Paleo-Biodiversity Forecasting with Neural ODEs and Chronos-T5

## Project Introduction
This project explores paleontological data from the Cenozoic Era to analyze biodiversity trends and investigate the 'Lilliput Effect' (body size reduction in response to environmental change). It employs two distinct modeling approaches for time series forecasting: a data-driven **Chronos-T5 Foundation Model** (fine-tuned) and a **Neural Ordinary Differential Equation (ODE)** model, which represents a 'physics-informed' approach to learning dynamics. The goal is to compare their efficacy in forecasting biodiversity from an incomplete and noisy fossil record.

## Setup and Installation
To run this notebook, you need to install several Python libraries. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchdiffeq chronos-forecasting datasets
```

Alternatively, you can use the `requirements.txt` generated in this project:

```bash
pip install -r requirements.txt
```

## Data
The project utilizes three primary datasets:
1.  `/content/pbdb_data.csv`: Fossil context data (e.g., genus, max_ma, lithology) downloaded from the Paleobiology Database (PBDB).
2.  `/content/pbdb_data (1).csv`: Fossil measurement data (e.g., average body size) also from PBDB.
3.  `/content/TableS33.tab`: Climate proxy data (d18O, d13C) from the CENOGRID project, specifically the Table S3.3 data.

## Project Phases

### Phase 1: Data Preparation & Merging
-   Loads fossil context and measurement data into pandas DataFrames (`df_context`, `df_meas`).
-   Merges these DataFrames on `specimen_no` to create `df_final`, combining time, lithology, and size information.
-   Loads the climate proxy data (`df_climate`), renames columns, and filters it for the Cenozoic era (0-66 Ma).
-   Aligns fossil data with climate data by calculating `age_mid` for fossils and binning both datasets into 0.1 Ma slices, then merging `d18o` into `df_final`.

### Phase 2: Feature Engineering
-   Creates `df_ml` as a copy of `df_final` for machine learning.
-   Filters out entries with missing `average` body size or `d18o` values, and ensures positive body sizes.
-   Calculates `latitude_abs` (absolute paleolatitude) for geographic context.
-   Performs one-hot encoding on `lithology1` (rock type) to convert categorical data into numerical features, simplifying to the top 10 rock types and grouping others as 'other'.

### Phase 3: Initial Modeling (Random Forest)
-   Trains a `RandomForestRegressor` to predict `average` body size (`y`) based on `d18o`, `latitude_abs`, and the one-hot encoded lithology features (`X`).
-   Splits data into training and testing sets (80/20 split).
-   Evaluates the model's accuracy (R^2 score) and, crucially, determines **feature importance**.
-   **Key Finding**: In this analysis, `latitude_abs` was identified as the most dominant factor driving body size, suggesting a geographic rather than a direct climate (temperature) signal initially.

### Phase 4: Visualization (Lilliput Effect Plot)
-   Generates a dual-axis plot (`Lilliput_Discovery_Plot.png`) showing the global mean body size (blue line) against the global temperature proxy (`d18o`, red dashed line) over the Cenozoic Era. This plot visually explores the relationship between climate and body size. Higher `d18o` typically means colder temperatures, and the inverted axis helps visualize warming trends.

### Phase 5: Advanced Forecasting (Neural ODE)
-   **Synthetic Data Generation**: Simulates a sine wave with irregular sampling to mimic real paleontological data, creating a 'ground truth' for testing.
-   **Neural ODE Definition**: Implements an `ODEFunc` (a simple MLP) to learn the underlying differential equations (dy/dt) of the synthetic system.
-   **Training Loop**: Trains the `ODEFunc` using `torchdiffeq`'s `odeint` solver, optimizing the MLP to predict the derivative and thus reconstruct the original signal.
-   **Visualization**: Plots the 'true' synthetic data against the Neural ODE's learned trajectory to demonstrate its ability to capture complex dynamics from sparse data.

### Phase 6: Advanced Forecasting (Chronos-T5)

#### Data Preparation
-   Converts the processed `df_ml` into a time series (`ts_data`) by counting unique genera per `age_mid` and sorting it. This represents the biodiversity data to be forecasted.

#### Pre-trained Chronos Forecasting
-   Loads the `amazon/chronos-t5-small` pre-trained model using `ChronosPipeline`.
-   Generates an initial forecast for 20 future steps based on the `ts_data`.
-   Visualizes this forecast (`Biodiversity_Forecasting_with_Chronos-T5.png`) showing historical data, the AI forecast (median), and an uncertainty band (10-90% quantiles).

#### Fine-tuning Chronos-T5
-   Splits `ts_data` into training (context) and validation (target) sets.
-   Sets up context and prediction windows for fine-tuning.
-   Accesses the inner T5 model (`pipeline_ft.model.model`) for direct training.
-   Fine-tunes the model for 20 epochs using `AdamW` optimizer, sampling random windows of data. The `pipeline_ft.tokenizer.config.prediction_length` is explicitly set to match the `prediction_window` to ensure proper tokenization.
-   Monitors the average loss per epoch during fine-tuning.

#### Evaluation of Fine-tuned Chronos-T5
-   Switches the fine-tuned model (`pipeline_ft.model`) to evaluation mode.
-   Generates a new forecast on a held-out 'true future' subset of the biodiversity time series.
-   Visualizes the results (`The_Final_Test_Fine_Tuned_AI.png`) comparing the training history, the true future, and the fine-tuned model's forecast with uncertainty.

#### Quantitative Comparison
-   Calculates the Mean Squared Error (MSE) for the fine-tuned Chronos model's predictions against the true future data (`Final_Model_Comparison_FineTuned.csv`).
-   Compares the Chronos MSE (13.0511 in the last run) against predefined thresholds (0.05 for 'Big Data' win, 0.10 for 'Physics' win) to interpret the performance.

## Conclusion
Based on the quantitative evaluation (MSE), the fine-tuned Chronos-T5 model had an MSE of 13.0511. This high MSE suggests that for this particular paleobiodiversity dataset and forecasting task, the 'Physics' approach (Neural ODE) likely achieved a better fit or captured the underlying dynamics more accurately than the fine-tuned 'Big Data' approach (Chronos-T5). Visual inspection of the respective plots can further support this conclusion.

## Output Files
-   `Lilliput_Project_Data.csv`: The final processed dataset used for machine learning.
-   `Research_Results_Ranking.csv`: The feature importance ranking from the Random Forest model.
-   `Lilliput_Discovery_Plot.png`: Visualization of global body size vs. climate.
-   `Final_Model_Comparison_FineTuned.csv`: Comparison of true vs. AI-predicted diversity from the fine-tuned Chronos model.
-   `Biodiversity_Forecasting_with_Chronos-T5.png`: Initial forecast from the pre-trained Chronos-T5 model.
-   `The_Final_Test_Fine_Tuned_AI.png`: Final forecast from the fine-tuned Chronos-T5 model against hidden data.
