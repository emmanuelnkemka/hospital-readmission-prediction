import requests

url = 'http://localhost:9696/predict'

patient = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[80-90)",
    "admission_type_id": "1",
    "discharge_disposition_id": "1",
    "admission_source_id": "7",
    "time_in_hospital": 3,
    "medical_specialty": "unknown",
    "num_lab_procedures": 37,
    "num_procedures": 1,
    "num_medications": 14,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "diag_1": "53",
    "diag_2": "462",
    "diag_3": "53",
    "number_diagnoses": 7,
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "insulin": "Steady",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "Yes"
}

response = requests.post(url, json=patient).json()

print(response)