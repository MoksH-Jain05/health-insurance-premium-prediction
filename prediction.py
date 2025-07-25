from joblib import load
import pandas as pd

model_young = load("artifacts\model_young.joblib")
model_rest = load("artifacts\model_rest.joblib")
scaler_young = load("artifacts\scaler_young.joblib")
scaler_rest = load("artifacts\scaler_rest.joblib")
scaler_score = load("artifacts\scaler_score.joblib")
expected_columns = load("artifacts\expected_col.joblib")
def cal_risk_score(col):
    risk_scores = {
        'diabetes': 6,
        'heart disease': 8,
        'high blood pressure': 6,
        'thyroid': 5,
        'none': 0,
        'no disease': 0
    }
    diseases = col[0].lower().replace('_',' ').split('&')
    total_risk_score = sum(risk_scores[disease.strip()] for disease in diseases)
    normalized_score = scaler_score.transform([[total_risk_score]])
    return normalized_score


def preprocess_input(input_dict,expected_columns):
    input = {key.lower().replace(' ','_').strip():value if type(value) ==int else value.replace(" ",'_') for key,value in input_dict.items()}
    df = pd.DataFrame([input])
    df.rename(columns={'income_in_lakhs': 'income_lakhs'}, inplace=True)
    df_encoded = pd.get_dummies(df,columns = ['gender','region','marital_status','bmi_category','smoking_status','employment_status'],dtype = int)
    df_encoded['normalized_score'] = cal_risk_score(df_encoded['medical_history'].values)
    df_encoded['insurance_plan'] = df_encoded['insurance_plan'].map({"Bronze":1,"Silver":2,"Gold":3})
    for col in df_encoded.columns:
        if col not in expected_columns:
            df_encoded.drop(col,axis = 1,inplace = True)

    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_input,model = handle_scaling(df_encoded['age'].iloc[0],df_encoded)
    df_reordered = df_input[expected_columns]
    return df_reordered,model

def handle_scaling(age,df):
    if age<=25:
        scaler_object = scaler_young
        model = model_young
    else:
        scaler_object = scaler_rest
        model = model_rest
    col_to_scale =scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']
    df['income_level']= None
    df[col_to_scale] = scaler.transform(df[col_to_scale])
    df.drop('income_level', axis=1, inplace=True)
    return df , model
def predict(input_dict):
    df,model = preprocess_input(input_dict,expected_columns)
    print(df.values)
    prediction = model.predict(df)

    return int(prediction[0])