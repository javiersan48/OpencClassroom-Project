from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
import pandas as pd
import pickle
import uvicorn
import requests
from fastapi.responses import JSONResponse
import shap
import numpy as np
from fastapi.responses import HTMLResponse

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

def convert_columns_to_numeric(df):
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    
    for column in non_numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    return df



@app.get("/data_explore")
def read_explore_csv(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    df = pd.read_csv(r'P7_Data_Dashboard_explore.csv') 
    df = df.fillna('')               
    data = df.to_dict(orient="records")
    return data

@app.get("/data_predict")
def read_predict_csv(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    df = pd.read_csv(r'P7_Data_Dashboard_predict.csv')
    df = df.fillna('')
    data = df.to_dict(orient="records")
    return data

@app.get("/image_logo")
async def get_logo_image(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    image_path = r"logo_pret_a_depenser.PNG"
    image = Image.open(image_path)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return StreamingResponse(image_bytes, media_type="image/png")

@app.get("/image_cover")
async def get_cover_image(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    image_path = r"PhotoJ.PNG"
    image = Image.open(image_path)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return StreamingResponse(image_bytes, media_type="image/png")

@app.get("/image_lgbm")
async def get_lgbm_image(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    image_path = r"LightGBM.png"
    image = Image.open(image_path)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return StreamingResponse(image_bytes, media_type="image/png")

@app.get("/image_credit")
async def get_credit_image(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    image_path = r"dreamcredit.PNG"
    image = Image.open(image_path)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return StreamingResponse(image_bytes, media_type="image/png")



model_url = 'https://github.com/javiersan48/OpencClassroom-Project/raw/master/LightGBM_with_threshold.pkl'
response = requests.get(model_url)
model_with_threshold = pickle.loads(response.content)
model = model_with_threshold['model']


df2 = pd.read_csv(r'P7_Data_Dashboard_predict.csv')
df2.drop(columns=['Unnamed: 0'], inplace=True)
df2.set_index('SKIDCURR', inplace=True)
df2 = df2.fillna('')
df2 = convert_columns_to_numeric(df2)
# Créer un explainer SHAP avec le modèle entraîné
explainer = shap.Explainer(model)

@app.get("/global_shap_values")
async def get_shap_values(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    
    # Calculer les valeurs SHAP pour les données d'entrée
    shap_values = explainer.shap_values(df2)
    shap_values_list = [values.tolist() for values in shap_values]
    return JSONResponse(content={"shap_values": shap_values_list}, media_type="application/json")

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
    with open(r"DataDrift.html", "r", encoding="utf-8") as report:
        html_content = report.read()
        return html_content
    
