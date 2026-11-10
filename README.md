# Hospital Readmission Prediction

## Project Overview

Hospital readmissions are an important healthcare challenge because patients who return shortly after discharge may require additional treatment, increase healthcare utilization, and indicate unresolved clinical or care-transition issues.

This project builds a machine-learning model to predict whether a patient will be readmitted to the hospital within 30 days of discharge.

The project uses historical hospital encounter data from the **Diabetes 130-US Hospitals dataset**. The final model is exposed through a Flask web service, served with Gunicorn, and containerized with Docker.

The objective is to demonstrate the complete machine-learning workflow covered in ML Zoomcamp:

- data preparation and cleaning
- exploratory data analysis
- feature importance analysis
- model training
- hyperparameter tuning
- model evaluation
- exporting training logic to a Python script
- serving predictions through a web API
- containerizing the application with Docker

---

## Problem Statement

The goal of this project is to predict:

> **Will a patient be readmitted to the hospital within 30 days?**

The original dataset contains three readmission outcomes:

- `<30` - patient was readmitted within 30 days
- `>30` - patient was readmitted after more than 30 days
- `NO` - patient was not readmitted

For this project, the target was converted into a binary classification problem:

```text
1 = readmitted within 30 days
0 = not readmitted within 30 days
```

After preprocessing and retaining one encounter per patient, approximately **9% of patients** belong to the positive class.

Because the target is imbalanced, accuracy alone is not sufficient for evaluating the models. The primary model-selection metric is **ROC AUC**, supported by precision, recall, F1-score, and confusion-matrix analysis.

## Who would use this

-   Hospitals are financially penalized by CMS for 30-day readmissions, so  there’s a direct money reason to predict them

-   A discharge planner can make a limited number of follow-up calls per week

-   The model ranks today’s discharges so those calls go to the highest-risk patients first

-   It’s a triage tool for a scarce resource, not a clinical diagnosis

---

## Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository.

The original dataset contains:

- 101,766 hospital encounters
- 50 original columns
- patient demographics
- diagnoses
- medications
- previous hospital utilization
- admission and discharge information
- laboratory and procedure information
- readmission outcome

Dataset source:

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

### Downloading the Data

Create the data directory:

```bash
mkdir -p data
```

Download the dataset:

```bash
cd data

curl -L -o diabetes.zip \
"https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

unzip diabetes.zip
```

The main dataset used by the project is:

```text
data/diabetic_data.csv
```

The accompanying mapping file is:

```text
data/IDS_mapping.csv
```

---

## Data Preparation

Several preprocessing steps are performed before modeling.

### Target Creation

The original `readmitted` variable is converted into the binary target `readmitted_30`.

```python
df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
```

### Missing Values

The dataset uses `?` to represent missing information. These values are converted to proper missing values using NumPy.

```python
df = df.replace("?", np.nan)
```

Columns with extremely high missingness are removed, including:

- `weight` (96.9% missing)
- `max_glu_serum` (94.7% missing)
- `A1Cresult` 83% of patients had no A1C test recorded. Note that these are not
  missing values in the usual sense: the raw file stores `"None"`, meaning the test
  was not ordered, and pandas reads that string as missing. Whether a test was
  ordered is itself potentially informative, so dropping this column is a
  limitation of the project rather than a clean win.
- `payer_code`(39.6%) a judgment call rather than a necessity. Retaining
  it as an explicit `"unknown"` category would also have been defensible.

Remaining missing categorical values are replaced with `"unknown"`.

### Discharge Exclusions

Patients whose discharge disposition indicated death or hospice care were excluded because these patients could not realistically experience a subsequent hospital readmission.

### One Encounter Per Patient

Some patients appear multiple times in the original dataset.

To reduce the possibility of patient-level leakage between training and validation data, encounters are ordered by `encounter_id` and only the first recorded encounter for each patient is retained.

After this step, the modeling dataset contains approximately **69,990 unique patients**.

### Identifier and Leakage Removal

The following columns are removed before modeling:

- `encounter_id`
- `patient_nbr`
- original `readmitted` variable

The original `readmitted` column must not be used as a feature because the binary target `readmitted_30` is derived directly from it.

### Constant Columns

Features containing only one unique value are removed because they provide no predictive information.

---

## Exploratory Data Analysis

The notebook examines several relationships between patient characteristics and 30-day hospital readmission.

### Target Distribution

After preprocessing, approximately:

```text
91% = not readmitted within 30 days
9%  = readmitted within 30 days
```

This class imbalance strongly influenced the evaluation strategy.

### Age

Readmission generally increases with age.

For example:

```text
Age 0-10    ≈ 2%
Age 60-70   ≈ 9%
Age 70-80   ≈ 10%
Age 80-90   ≈ 11%
```

Older patients therefore appear to have greater readmission risk.

### Length of Hospital Stay

Readmission rates generally increase as the length of the hospital stay increases.

Examples:

```text
1 day      ≈ 6.5%
4 days     ≈ 8.9%
6 days     ≈ 10.9%
8 days     ≈ 12.3%
```

### Previous Inpatient Visits

Previous inpatient utilization was one of the strongest numerical signals in the dataset.

Examples:

```text
0 previous admissions   ≈ 8.1%
1 previous admission    ≈ 12.9%
2 previous admissions   ≈ 18.5%
3 previous admissions   ≈ 23.8%
```

Patients with repeated prior hospitalizations show substantially higher readmission rates.

### Emergency Department Visits

Previous emergency department utilization also shows a strong relationship with readmission:

```text
0 visits   ≈ 8.7%
1 visit    ≈ 11.3%
2 visits   ≈ 13.4%
3+ visits  ≈ 16.3%
```

Outpatient visits showed a substantially weaker relationship with the target.

---

## Feature Importance

### Numerical Correlation

Among the numerical features, `number_inpatient` had the strongest positive linear relationship with 30-day readmission.

Selected correlations:

```text
number_inpatient       0.1003
time_in_hospital       0.0558
number_diagnoses       0.0456
num_medications        0.0362
num_lab_procedures     0.0321
number_emergency       0.0280
number_outpatient      0.0086
num_procedures        -0.0003
```

These correlations are relatively small, which suggests that readmission is influenced by multiple interacting factors rather than a single strong linear predictor.

### Categorical Feature Importance

Mutual information was used to evaluate categorical variables.

The strongest categorical features included:

```text
diag_1               0.009842
diag_3               0.008063
diag_2               0.007541
medical_specialty    0.001682
age                  0.001408
```

The diagnosis variables contained the strongest categorical signal.

---

## Train, Validation and Test Split

The dataset is divided into:

```text
60% training
20% validation
20% test
```

Stratification is used so that the approximately 9% readmission rate remains similar across all three datasets.

The split uses:

```python
random_state=1
```

for reproducibility.

---

## Models

Validation ROC AUC during model selection: 0.647. The final model, retrained on the combined training and validation data, scored 0.6595 on the held-out test set — a figure never used for any tuning decision.

Three machine-learning models were trained and compared.

### Logistic Regression

The baseline Logistic Regression model achieved:

```text
Validation ROC AUC ≈ 0.620
```

At the default classification threshold of `0.50`, recall was extremely low because the model rarely predicted the minority class.

Threshold analysis showed that lowering the threshold substantially increased recall.

The best F1-score for Logistic Regression was obtained at approximately:

```text
threshold ≈ 0.12
precision ≈ 0.151
recall    ≈ 0.359
F1        ≈ 0.212
```

### Decision Tree

The Decision Tree was tuned using combinations of:

- `max_depth`
- `min_samples_leaf`

The best validation result was approximately:

```text
ROC AUC ≈ 0.620
max_depth = 8
min_samples_leaf = 50
```

### Random Forest

Random Forest produced the strongest validation performance.

Parameters explored included:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`

The best configuration was:

```text
n_estimators = 200
max_depth = None
min_samples_leaf = 5
```

Best validation performance:

```text
ROC AUC ≈ 0.647
```

### Model Comparison

| Model | Validation ROC AUC |
|---|---:|
| Logistic Regression | 0.620 |
| Decision Tree | 0.620 |
| Random Forest | **0.647** |

Random Forest was selected as the final model.

The final model was retrained on the combined training and validation data
(55,992 patients) and evaluated once on the held-out test set:

**Test ROC AUC: 0.6595**

This figure was never used for any tuning decision. It is higher than the
validation score because the final model was trained on a third more data.

---

## Project Structure

```text
hospital-readimission-prediction/
│
├── data/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
│
├── hospital-readimission.ipynb
├── train.py
├── predict.py
├── predict-test.py
├── model.bin
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## Environment Setup

This project was developed using **Python 3.11**.

Clone the repository:

```bash
git clone https://github.com/emmanuelnkemka/hospital-readmission-prediction.git
cd hospital-readmission-prediction
```

Create a virtual environment:

```bash
python3.11 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Notebook

Activate the environment:

```bash
source venv/bin/activate
```

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebook.ipynb
```

The notebook contains:

- data loading
- data preparation
- missing-value analysis
- EDA
- feature importance analysis
- train/validation/test split
- Logistic Regression
- Decision Tree
- Random Forest
- hyperparameter tuning
- model comparison

The notebook is designed to run from top to bottom from a fresh kernel.

---

## Train the Final Model

Run:

```bash
python train.py
```

The training script preprocesses the data, trains the final model, and saves the model artifact as:

```text
model.bin
```

---

## Run the Prediction Service Locally

Activate the environment and run:

```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```

The API will be available at:

```text
http://localhost:9696
```

The prediction endpoint is:

```text
POST /predict
```

Test it from another terminal:

```bash
python predict-test.py
```

Example response:

```python
{
    "readmission_probability": 0.08722504939094483,
    "readmit": False
}
```

This indicates that the model estimated approximately an **8.7% probability of 30-day readmission** for the example patient.

---

## Docker

The prediction service is fully containerized.

### Build the Image

From the project root:

```bash
docker build -t readimission-prediction .
```

### Run the Container

```bash
docker run -it --rm -p 9696:9696 readimission-prediction
```

The service will be exposed on:

```text
http://localhost:9696
```

### Test the Containerized Service

From another terminal:

```bash
python predict-test.py
```

Example output:

```python
{
    "readmission_probability": 0.08722504939094483,
    "readmit": False
}
```

This confirms that the model can be loaded and queried successfully from inside the Docker container.

---

## Reproducibility

To reproduce the project from a clean environment:

```bash
git clone https://github.com/emmanuelnkemka/hospital-readmission-prediction.git
cd hospital-readmission-prediction

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Download the dataset if it is not included in the repository.

Then execute:

```bash
python train.py
```

Start the prediction service:

```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```

Or build and run the Docker image:

```bash
docker build -t readimission-prediction .
docker run -it --rm -p 9696:9696 readimission-prediction
```

---

## Technologies Used

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Flask
- Gunicorn
- Docker
- Jupyter Notebook
- Git
- GitHub

---

## Limitations

The model's ROC AUC is approximately 0.65, which indicates moderate predictive performance rather than production-grade clinical performance.

Several limitations should be considered:

- the dataset is historical
- the positive class is highly imbalanced
- diagnosis codes are high-cardinality categorical variables
- potentially useful variables contain substantial missing data
- only the first observed encounter for each patient is retained
- the model has not been clinically validated
- the model should not be used for real patient-care decisions

This project is intended for educational machine-learning purposes.

---

## Future Improvements

Possible improvements include:

- more advanced diagnosis-code feature engineering
- class-weight tuning
- additional threshold optimization
- cross-validation
- probability calibration
- additional ensemble methods
- cloud deployment
- monitoring model performance over time

---

## Author

**Emmanuel Nkemka**

ML Zoomcamp 2026 Midterm Project