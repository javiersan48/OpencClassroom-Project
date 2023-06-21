from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import pandas as pd
import pickle
import io
import boto3
import uvicorn
from fastapi.responses import JSONResponse
import shap
import numpy as np
from io import BytesIO
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse

app = FastAPI()
security = HTTPBasic()

# Vérification des informations d'identification
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "Openclassroom"
    correct_password = "Jerome_S"
    if (
        credentials.username != correct_username
        or credentials.password != correct_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Basic"},
        )

AWS_ACCESS_KEY = 'AKIA6FATDVCIIHLH7QVS'
AWS_SECRET_ACCESS_KEY = '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'
BUCKET_NAME = 'projet7'
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)



@app.get("/data_explore")
def read_explore_csv(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    response = s3.get_object(Bucket= BUCKET_NAME, Key= 'P7_Data_Dashboard_explore.csv')
    file_content = response['Body'].read().decode('utf-8')
    return StreamingResponse(io.StringIO(file_content), media_type='text/csv')

@app.get("/data_predict")
def read_predict_csv(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    response = s3.get_object(Bucket= BUCKET_NAME, Key= 'P7_Data_Dashboard_predict.csv')
    file_content = response['Body'].read().decode('utf-8')
    return StreamingResponse(io.StringIO(file_content), media_type='text/csv')

# @app.get("/image_logo")
# async def get_logo_image(credentials: HTTPBasicCredentials = Depends(security)):
#     verify_credentials(credentials)
#     s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
#     response = s3.get_object(Bucket=BUCKET_NAME, Key='logo_pret_a_depenser.PNG')
#     file_content = response['Body'].read()
#     return file_content

@app.get("/image_logo")
async def get_logo_image(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    response = s3.get_object(Bucket=BUCKET_NAME, Key='logo_pret_a_depenser.PNG')
    file_content = response['Body'].read()
    return StreamingResponse(BytesIO(file_content), media_type="image/png")


@app.get("/image_cover")
async def get_cover_image(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    response = s3.get_object(Bucket=BUCKET_NAME, Key='PhotoJ.PNG')
    file_content = response['Body'].read()
    return StreamingResponse(BytesIO(file_content), media_type="image/png")

@app.get("/image_lgbm")
async def get_lgbm_image(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    response = s3.get_object(Bucket = BUCKET_NAME, Key='LightGBM.png')
    file_content = response['Body'].read()
    return StreamingResponse(BytesIO(file_content), media_type="image/png")

@app.get("/image_credit")
async def get_credit_image(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    response = s3.get_object(Bucket = BUCKET_NAME, Key='dreamcredit.PNG')
    file_content = response['Body'].read()
    return StreamingResponse(BytesIO(file_content), media_type="image/png")



model_path = r'C:\Users\jerom\Documents\Notebooks\Openclassroom\P7\LightGBM_with_threshold.pkl'
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
response = s3.get_object(Bucket = BUCKET_NAME, Key = 'LightGBM_with_threshold.pkl')
model_content = response['Body'].read()
model = pickle.loads(model_content)    
model = model['model']

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
response = s3.get_object(Bucket= BUCKET_NAME, Key= 'P7_Data_Dashboard_predict.csv')
file_content = response['Body'].read().decode('utf-8')
df2 = pd.read_csv(io.StringIO(file_content))


# df2 = pd.read_csv(r'C:\Users\jerom\Desktop\DataZ\P7_Data_Dashboard_predict.csv')
df2.drop(columns=['Unnamed: 0'], inplace=True)
df2.set_index('SKIDCURR', inplace=True)

# Créer un explainer SHAP avec le modèle entraîné
explainer = shap.Explainer(model)

# @app.get("/global_shap_values")
# async def get_shap_values(credentials: HTTPBasicCredentials = Depends(security)):
#     verify_credentials(credentials)
    
#     # Calculer les valeurs SHAP pour les données d'entrée
#     shap_values = explainer.shap_values(df2)
#     return JSONResponse(content={"shap_values": shap_values.tolist()}, media_type="application/json")

# def get_local_shap_values(num_client):
#     # Obtenir les données utilisateur correspondant au numéro client
#     user = df2[df2.index == int(num_client)]
    
#     # Calculer les valeurs SHAP pour les données utilisateur
#     shap_values = explainer.shap_values(user)
    
#     # Obtenir la valeur attendue (expected value)
#     expected_value = explainer.expected_value[0]
    
#     return shap_values, expected_value


@app.get("/global_shap_values")
async def get_shap_values(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    
    # Calculer les valeurs SHAP pour les données d'entrée
    shap_values = explainer.shap_values(df2)
    shap_values_list = [values.tolist() for values in shap_values]
    return JSONResponse(content={"shap_values": shap_values_list}, media_type="application/json")

# def get_local_shap_values(num_client):
#     # Obtenir les données utilisateur correspondant au numéro client
#     user = df2[df2.index == int(num_client)]
    
#     # Calculer les valeurs SHAP pour les données utilisateur
#     shap_values = explainer.shap_values(user)
    
#     # Obtenir la valeur attendue (expected value)
#     expected_value = explainer.expected_value[0]
    
#     return shap_values, expected_value

# @app.get("/local_shap_values/{num_client}")
# async def get_shap_values_by_client(num_client: str, credentials: HTTPBasicCredentials = Depends(security)):
#     verify_credentials(credentials)
    
#     # Obtenir les valeurs SHAP pour le numéro client spécifié
#     shap_values, expected_value = get_local_shap_values(num_client)
    
#     return JSONResponse(content={"shap_values": shap_values.tolist(), "expected_value": expected_value}, media_type="application/json")

# def get_probabilities(num_client):
#     # Obtenir les données utilisateur correspondant au numéro client
#     user = df2[df2.index == int(num_client)]
    
#     # Calculer les probabilités de prédiction pour les données utilisateur
#     probas_user = model.predict_proba(user)
    
#     # Créer un dictionnaire des probabilités arrondies
#     probabilities = dict(zip(model.classes_, np.round(probas_user[0], 3)))
    
#     return probabilities

# @app.get("/probabilities/{num_client}")
# async def get_probabilities_by_client(num_client: str, credentials: HTTPBasicCredentials = Depends(security)):
#     verify_credentials(credentials)
    
#     # Obtenir les probabilités de prédiction pour le numéro client spécifié
#     probabilities = get_probabilities(num_client)
    
#     return JSONResponse(content={"probabilities": probabilities}, media_type="application/json")
def get_local_shap_values(num_client):
    # Obtenir les données utilisateur correspondant au numéro client
    user = df2[df2.index == int(num_client)]
    
    # Calculer les valeurs SHAP pour les données utilisateur
    shap_values = explainer.shap_values(user)
    
    # Obtenir la valeur attendue (expected value)
    expected_value = explainer.expected_value[0]
    
    return shap_values, expected_value

@app.get("/local_shap_values/{num_client}")
async def get_shap_values_by_client(num_client: str, credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    
    # Obtenir les valeurs SHAP pour le numéro client spécifié
    shap_values, expected_value = get_local_shap_values(num_client)
    shap_values_list = [values.tolist() for values in shap_values]
    return JSONResponse(content={"shap_values": shap_values_list, "expected_value": expected_value}, media_type="application/json")

def get_probabilities(num_client):
    # Obtenir les données utilisateur correspondant au numéro client
    user = df2[df2.index == int(num_client)]
    
    # Calculer les probabilités de prédiction pour les données utilisateur
    probas_user = model.predict_proba(user)
    
    # Créer un dictionnaire des probabilités arrondies
    probabilities = dict(zip(model.classes_, np.round(probas_user[0], 3)))
    
    return probabilities

@app.get("/probabilities/{num_client}")
async def get_probabilities_by_client(num_client: str, credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    
    # Obtenir les probabilités de prédiction pour le numéro client spécifié
    probabilities = get_probabilities(num_client)
    
    return JSONResponse(content={"probabilities": probabilities}, media_type="application/json")
@app.get("/data_drift", response_class = HTMLResponse)
async def get_data_drift(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    response = s3.get_object(Bucket = BUCKET_NAME, Key='DataDrift.html')
    file_content = response['Body'].read().decode('utf-8')
    return file_content
    
if __name__ == '__main__':
    uvicorn.run("API Local:app", host='127.0.0.1', port=8000)
    #uvicorn app:app --reload
    #cd C:\Users\jerom\Desktop\chamsedine
    #uvicorn FastAPI_AWS:app --reload