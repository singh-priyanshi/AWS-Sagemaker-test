import os
import joblib
import numpy as np
import json
from io import StringIO
import pandas as pd
# Define model_fn to load the model
def model_fn(model_dir):
    """
    Load the model from the model_dir.
    SageMaker automatically loads model artifacts from the model.tar.gz file located in the S3 bucket.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

# Define input_fn to parse the incoming request
def input_fn(input_data, content_type):
    """
    Preprocess the input data before passing it to the model.
    content_type could be 'application/json', 'text/csv', etc.
    """
    if content_type == 'application/json':
        # For JSON input, parse the input as JSON
        
        return pd.DataFrame(json.loads(input_data))
    elif content_type == 'text/csv':
        # For CSV input, parse the input as a CSV
        return pd.read_csv(StringIO(input_data))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Define predict_fn to generate predictions
def predict_fn(input_data, model):
    """
    Perform prediction using the loaded model.
    """
    prediction = model.predict(input_data)
    return prediction

# Define output_fn to post-process and format the predictions
def output_fn(prediction, accept):
    """
    Format the output data into JSON or another format to return it to the client.
    """
    if accept == 'application/json':
        return json.dumps(prediction.tolist()), 'application/json'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
