import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# parameters
n_estimators = 200
max_depth = None
min_samples_leaf = 5
random_state = 1

output_file = 'model.bin'

# load and clean
print('loading data...')

df = pd.read_csv('data/diabetic_data.csv')

# build the target
df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)

# '?' is how this dataset encodes missing
df = df.replace('?', np.nan)

# patients who died or went to hospice could not be readmitted
discharge_exclude = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(discharge_exclude)]

# one row per patient — keep the earliest encounter
df = df.sort_values('encounter_id')
df = df.drop_duplicates(subset='patient_nbr', keep='first')

# these are category codes, not quantities
id_columns = [
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id'
]
for col in id_columns:
    df[col] = df[col].astype(str)

# drop columns that are mostly missing
columns_to_drop = [
    'weight',
    'max_glu_serum',
    'A1Cresult',
    'payer_code'
]
df = df.drop(columns=columns_to_drop)

# fill the remaining gaps
df['medical_specialty'] = df['medical_specialty'].fillna('unknown')
for col in ['race', 'diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna('unknown')

# drop columns with a single value
constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
df = df.drop(columns=constant_columns)

# identifiers and the original target leak or add nothing
df = df.drop(columns=['encounter_id', 'patient_nbr', 'readmitted'])

print(f'data ready: {df.shape[0]} rows, {df.shape[1]} columns')

# split
print('splitting...')

df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=random_state,
    stratify=df.readmitted_30
)

df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=random_state,
    stratify=df_full_train.readmitted_30
)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.readmitted_30.values
y_train = df_train.readmitted_30.values
y_val = df_val.readmitted_30.values
y_test = df_test.readmitted_30.values

del df_full_train['readmitted_30']
del df_train['readmitted_30']
del df_val['readmitted_30']
del df_test['readmitted_30']

print(f'train: {len(df_train)}  val: {len(df_val)}  test: {len(df_test)}')

# training


def train(df, y):
    dicts = df.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)

    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)
    return model.predict_proba(X)[:, 1]

# validation
print('training on train split...')

dv, model = train(df_train, y_train)
y_pred = predict(df_val, dv, model)

auc = roc_auc_score(y_val, y_pred)
print(f'validation auc: {auc:.4f}')

# final model
print('training final model on full training set...')

dv, model = train(df_full_train, y_full_train)

y_pred_test = predict(df_test, dv, model)
test_auc = roc_auc_score(y_test, y_pred_test)
print(f'test auc: {test_auc:.4f}')

# save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'model saved to {output_file}')