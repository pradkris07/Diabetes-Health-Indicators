from flask import Flask, render_template, request
import pandas as pd
import pickle
import yaml
from src.constants import *
from dotenv import load_dotenv
import mlflow
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
load_dotenv()

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{DAGSHUB_URL}/{REPO_OWNER}/{REPO_NAME}.mlflow')
# -------------------------------------------------------------------------------------

app = Flask(__name__)

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name,stages=["None"])
    print(latest_version)
    return latest_version[0].version if latest_version else None
    #return 5

def load_yaml():
    with open(MODEL_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

@app.route("/")
def home():
    data = load_yaml()
    name_list=[]
    for ind,val in enumerate(data["models"]):
        name_list.append(data["models"][ind]['name'])
    return render_template("healthform_flask.html", estimators=name_list, result=None)   # your html filename

@app.route("/predict", methods=["POST"])
def predict():
    # Collect all values from the form
    form_data = {
        "HighBP": request.form.get("High_BP"),
        "HighChol": request.form.get("High_Cholesterol"),
        "CholCheck": request.form.get("Cholesterol_Checked"),
        "BMI": request.form.get("BMI"),
        "Smoker": request.form.get("Smoker"),
        "Stroke": request.form.get("Had_Stroke"),
        "HeartDiseaseorAttack": request.form.get("Heart_Disease"),
        "PhysActivity": request.form.get("Physical_Activity"),
        "Fruits": request.form.get("Consume_Fruits"),
        "Veggies": request.form.get("Consume_Veggies"),
        "HvyAlcoholConsump": request.form.get("Drink_Heavily"),
        "AnyHealthcare": request.form.get("Subscribe_Healthcare"),
        "NoDocbcCost": request.form.get("NoDocbcCost"),
        "GenHlth": request.form.get("General_Health"),
        "MentHlth": request.form.get("Mental_Health"),
        "PhysHlth": request.form.get("Physical_Health"),
        "DiffWalk": request.form.get("Difficulty_Walking"),
        "Sex": request.form.get("Sex"),
        "Age": request.form.get("Age"),
        "Education": request.form.get("Education"),
        "Income": request.form.get("Income"),
        "Estimator": request.form.get("Estimator")
    }
    
    # Convert to DataFrame (one row)
    df = pd.DataFrame([form_data])
    print(df)
    
    # Convert to numeric where needed:
    df = df.astype({
        "HighBP": "int",
        "HighChol": "int",
        "CholCheck": "int",
        "BMI": "float",
        "Smoker": "int",
        "Stroke": "int",
        "HeartDiseaseorAttack": "int",
        "PhysActivity": "int",
        "Fruits": "int",
        "Veggies": "int",
        "HvyAlcoholConsump": "int",
        "AnyHealthcare": "int",
        "NoDocbcCost": "int",
        "GenHlth": "int",
        "MentHlth": "int",
        "PhysHlth": "int",
        "DiffWalk": "int",
        "Sex": "int",
        "Age": "int",
        "Education": "int",
        "Income": "int"
        })
    
    transformer = pickle.load(open(PREPROCSSING_OBJECT_FILE_NAME, 'rb'))
    df_without_est = df.drop(columns=['Estimator'], axis=1)
        
    transformed_df = transformer.transform(df_without_est)
    print("\n===== Form Submitted DataFrame =====")
    print(transformed_df)
    print("====================================\n")
    model_name = request.form["Estimator"]
    if model_name == 'Gradient Boosting':
        model_mlflow_name = GDB_MODEL_PATH.split('\\')[1].split('.')[0]
        model_version = get_latest_model_version(model_mlflow_name)
        model_uri = f'models:/{model_mlflow_name}/{model_version}'
        model = mlflow.pyfunc.load_model(model_uri)
        #model= pickle.load(open(GDB_MODEL_PATH, 'rb'))
    elif model_name == 'Random Forest':
        model_mlflow_name = RFC_MODEL_PATH.split('\\')[1].split('.')[0]
        model_version = get_latest_model_version(model_mlflow_name)
        model_uri = f'models:/{model_mlflow_name}/{model_version}'
        model = mlflow.pyfunc.load_model(model_uri)
        #model= pickle.load(open(RFC_MODEL_PATH, 'rb'))
    elif model_name == 'K Nearest Neighbour':
        model_mlflow_name = KNN_MODEL_PATH.split('\\')[1].split('.')[0]
        model_version = get_latest_model_version(model_mlflow_name)
        model_uri = f'models:/{model_mlflow_name}/{model_version}'
        model = mlflow.pyfunc.load_model(model_uri)
        #model= pickle.load(open(RFC_MODEL_PATH, 'rb'))
    elif model_name == 'Support Vectors Classifier':
        model_mlflow_name = SVC_MODEL_PATH.split('\\')[1].split('.')[0]
        model_version = get_latest_model_version(model_mlflow_name)
        model_uri = f'models:/{model_mlflow_name}/{model_version}'
        model = mlflow.pyfunc.load_model(model_uri)
        #model= pickle.load(open(RFC_MODEL_PATH, 'rb'))
    elif model_name == 'Gaussian NB Classifier':
        model_mlflow_name = GNB_MODEL_PATH.split('\\')[1].split('.')[0]
        model_version = get_latest_model_version(model_mlflow_name)
        model_uri = f'models:/{model_mlflow_name}/{model_version}'
        model = mlflow.pyfunc.load_model(model_uri)
        #model= pickle.load(open(RFC_MODEL_PATH, 'rb'))
    y_pred = model.predict(transformed_df)
    prediction = y_pred[0]
    
    return render_template("healthform_flask.html", result=prediction) 

    

    # (Optional) Convert to numeric where needed:
    # df = df.astype({
    #     "High_BP": "int",
    #     "High_Cholesterol": "int",
    #     ...
    # })

    # Return simple confirmation for now
    #return f"<h2>Received Data:</h2>{transformed_df.to_html(index=False)}"

if __name__ == "__main__":
    app.run(debug=True)
