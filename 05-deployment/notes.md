# Notes — Module 5: Deploying Machine Learning Models

## Course Overview
In this module, we focus on **Deploying Machine Learning Models**. In previous modules, our models lived entirely inside Jupyter notebooks. While notebooks are ideal for data analysis and experimentation, they cannot be directly consumed by production software systems. To make machine learning models useful, we must deploy them as reliable, reproducible web services. This module covers the end-to-end deployment pipeline: serializing models and feature transformers using **Pickle**, building web services with **Flask** (and modern **FastAPI**), running them via production **WSGI/ASGI** servers (**Gunicorn**, **Waitress**, **Uvicorn**), isolating dependencies using **Pipenv** (and modern **uv**), packaging everything into isolated **Docker** containers, and deploying the containerized application to the cloud (**AWS Elastic Beanstalk**, **Fly.io**).

---

## Explanation of Each Topic

### 1. Introduction to Model Deployment
A trained model inside a Jupyter notebook is inaccessible to other services. For example, in a telecom company, the marketing automation platform needs to score thousands of customers dynamically to determine who should receive promotional retention offers.
- **The Deployment Solution**: Deploy the model on a dedicated server and expose a network-accessible API endpoint.
- **Client-Server Architecture**: External services (clients) send HTTP requests containing customer feature data as JSON to the prediction endpoint. The prediction service passes the data through the model and returns predictions (e.g., churn probability) in the HTTP response.
- **The Decoupling Pipeline**: Deployment progressively decouples the model from our local machine:
  1. Notebook $\rightarrow$ Python Script & Saved Artifacts (Pickle)
  2. Local Script $\rightarrow$ Web Service (Flask/FastAPI)
  3. Development Server $\rightarrow$ Production WSGI/ASGI Server (Gunicorn/Uvicorn)
  4. System Python $\rightarrow$ Isolated Virtual Environment (Pipenv/uv)
  5. Local Host Machine $\rightarrow$ Containerized Environment (Docker)
  6. Local Container $\rightarrow$ Scalable Cloud Infrastructure (AWS Elastic Beanstalk / Fly.io)

### 2. Model Serialization with Pickle
During training, we create two crucial artifacts:
1. `DictVectorizer`: Encodes categorical features into numerical vectors.
2. `LogisticRegression` (or any classifier): The mathematical model weights and bias.

Both objects reside in volatile memory. Retraining every time a prediction is needed is slow, computationally expensive, and risks model drift due to random variations.

#### Saving Artifacts
We serialize both the vectorizer and the model into a single binary file using Python's built-in `pickle` library:
```python
import pickle

output_file = f'model_C={C}.bin'

with open(output_file, 'wb') as f_out:  # 'wb' indicates write-binary
    pickle.dump((dv, model), f_out)
```
> [!IMPORTANT]
> The `DictVectorizer` must always be saved alongside the model. The vectorizer "remembers" the exact feature mappings and one-hot encodings learned during training; without it, new data cannot be transformed into the correct feature space.

#### Loading Artifacts
To perform inference in a web service without executing training code:
```python
import pickle

with open('model_C=1.0.bin', 'rb') as f_in:  # 'rb' indicates read-binary
    dv, model = pickle.load(f_in)
```

> [!CAUTION]
> **Pickle Security Warning**: Never unpickle binary files from untrusted sources. Pickle can deserialize and execute arbitrary system commands, posing severe security vulnerabilities.

#### Converting Notebooks to Training Scripts
Moving code from a Jupyter notebook to a clean Python script (`train.py`) enables automated retraining, integration with CI/CD pipelines, and execution without needing a running Jupyter instance.

### 3. Web Services & Introduction to Flask
A **web service** allows distinct software systems to communicate over HTTP (Hypertext Transfer Protocol).

#### Core HTTP Methods for Machine Learning:
- `GET`: Used to retrieve resources or perform health-checks (e.g., `/ping`).
- `POST`: Used to submit data payloads to the server for processing (e.g., sending customer features to `/predict`).
- `PUT`: Creates or updates a resource at a specific URI.
- `DELETE`: Removes a resource from the server.

#### Building a Minimal Flask Service
```python
from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
```

#### Key Concepts:
- `@app.route(endpoint, methods)`: A Python decorator that registers an endpoint and its permitted HTTP methods.
- `host='0.0.0.0'`: Binds the server to all available network interfaces, allowing external traffic (including Docker containers) to reach the service. In contrast, `127.0.0.1` (`localhost`) only accepts requests originating from within the local machine.
- Testing via CLI:
  ```bash
  curl http://localhost:9696/ping
  ```

### 4. Serving Predictions with Flask
To build an actual prediction service, we load the saved model once at application startup and handle incoming POST requests containing customer attributes.

#### Prediction Service (`predict.py`):
```python
import pickle
from flask import Flask, request, jsonify

# Load serialized preprocessor and model once at startup
model_file = 'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()  # Parse incoming JSON body into Python dictionary

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]  # Churn probability
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
```

#### Client Test Script (`predict-test.py`):
```python
import requests

url = 'http://localhost:9696/predict'

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 41,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "bank_transfer_(automatic)",
    "monthlycharges": 79.85,
    "totalcharges": 3320.75
}

response = requests.post(url, json=customer).json()
print(response)

if response['churn']:
    print("Send promotional retention email.")
else:
    print("Do not send retention email.")
```

### 5. Production WSGI Servers (Gunicorn & Waitress)
Flask's built-in development server is single-threaded, lacks concurrency management, and explicitly warns against production use.

A **WSGI (Web Server Gateway Interface)** server manages multi-process request handling, timeouts, connection pooling, and worker recycling.

#### Production Deployment with Gunicorn (Linux / macOS):
```bash
pip install gunicorn
gunicorn --bind=0.0.0.0:9696 predict:app
```
- `--bind=0.0.0.0:9696`: Network address and listening port.
- `predict:app`: Refers to the `app` instance inside the `predict.py` module.

#### Production Deployment with Waitress (Windows):
Gunicorn relies on Unix-specific system calls not present in Windows. On Windows machines, **Waitress** is the standard pure-Python WSGI server:
```bash
pip install waitress
waitress-serve --listen=0.0.0.0:9696 predict:app
```

### 6. Dependency & Virtual Environment Management (Pipenv & uv)
Installing packages globally with `pip install` risks breaking older projects when shared libraries are updated (version conflicts).

#### Pipenv
**Pipenv** manages both package installation and virtual environments, generating two key files:
1. `Pipfile`: Declares high-level dependencies and required Python version.
2. `Pipfile.lock`: Contains precise version pins for all transitive dependencies alongside cryptographic SHA-256 hashes, ensuring deterministic builds across environments.

#### Essential Pipenv Commands:
- `pip install pipenv`: Install the CLI tool.
- `pipenv install numpy scikit-learn flask gunicorn`: Create environment and install packages.
- `pipenv install --dev awsebcli`: Install development-only dependencies.
- `pipenv shell`: Activate the isolated environment shell.
- `pipenv run <command>`: Execute a command inside the virtual environment without spawning a subshell.
- `pipenv install --deploy --system`: Installs packages from the lockfile directly into the system Python (ideal inside Docker containers).

#### Modern Alternative: `uv`
The Python ecosystem increasingly uses **`uv`**, an extremely fast Rust-based package and project manager developed by Astral:
- `uv init`: Initialize a new project (`pyproject.toml`).
- `uv add scikit-learn fastapi uvicorn`: Add locked dependencies into `.venv`.
- `uv sync`: Synchronize environment with `uv.lock`.
- `uv run <command>`: Run commands inside the managed virtual environment.

### 7. Environment Isolation with Docker
While virtual environments isolate Python packages, they do not isolate system-level C libraries, operating system differences, or the installed Python version itself. A service that runs on Ubuntu with Python 3.10 may fail on CentOS with Python 3.8.

**Docker** solves this by packaging the operating system, system libraries, Python runtime, application code, and model artifacts into a portable **Docker Image**. A running instance of an image is a **Docker Container**.

#### Writing a Production `Dockerfile`:
```dockerfile
# 1. Base image with minimal Debian runtime
FROM python:3.8.12-slim

# 2. Install package manager inside the container
RUN pip install pipenv

# 3. Create and set working directory
WORKDIR /app

# 4. Copy dependency specifications first (optimizes Docker layer caching)
COPY ["Pipfile", "Pipfile.lock", "./"]

# 5. Install locked dependencies into system environment without sub-virtualenv
RUN pipenv install --system --deploy

# 6. Copy application code and model artifacts
COPY ["predict.py", "model_C=1.0.bin", "./"]

# 7. Expose documentation port
EXPOSE 9696

# 8. Start production WSGI server
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
```

#### Building and Running Containers:
```bash
# Build image tagged as 'churn-prediction'
docker build -t churn-prediction .

# Run container with port forwarding and terminal attachment
docker run -it --rm -p 9696:9696 churn-prediction:latest
```
- `-t`: Assigns a name tag to the built image.
- `-it`: Keeps standard input/output open to interact with the container.
- `--rm`: Automatically removes the container upon stopping.
- `-p 9696:9696`: Maps host port `9696` to container port `9696` (`host_port:container_port`).

### 8. Cloud Deployment (AWS Elastic Beanstalk & Alternatives)
To make the service accessible over the internet with automated scaling and high availability, we deploy the containerized application to the cloud.

#### AWS Elastic Beanstalk (EB)
AWS Elastic Beanstalk is a Platform-as-a-Service (PaaS) that handles provisioning, load balancing, health monitoring, and horizontal autoscaling:
1. **Install EB CLI**:
   ```bash
   pipenv install awsebcli --dev
   ```
2. **Initialize Environment**:
   ```bash
   eb init -p docker -r eu-north-1 churn-serving
   ```
3. **Local Testing via EB**:
   ```bash
   eb local run --port 9696
   ```
4. **Deploy to Cloud**:
   ```bash
   eb create churn-serving-env
   ```
5. **Access Endpoint**: EB provisions a public domain URL (e.g., `http://churn-serving-env.eba-xxx.region.elasticbeanstalk.com/predict`).
6. **Clean Up**:
   ```bash
   eb terminate churn-serving-env
   ```

#### Alternative Cloud Deployment Targets:
- **Fly.io**: Seamless global container hosting via `fly launch` and `fly deploy`.
- **AWS App Runner / Google Cloud Run**: Serverless container execution with zero-scale capability.
- **Heroku**: PaaS deployment using `Procfile` and Git-based releases.

### 9. Modern Deployment Trends: FastAPI & Pydantic
Modern machine learning engineering frequently pairs **FastAPI** with **Pydantic**:
- **FastAPI**: An asynchronous ASGI framework offering native async execution, automatic data validation, and auto-generated Swagger UI docs (`/docs`).
- **Pydantic**: Enforces strict schema definitions and rejects unpermitted inputs (`extra="forbid"`):
```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI()

class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gender: str
    seniorcitizen: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0)
    monthlycharges: float
    totalcharges: float

class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool

@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    X = dv.transform([customer.model_dump()])
    prob = float(model.predict_proba(X)[0, 1])
    return PredictResponse(churn_probability=prob, churn=prob >= 0.5)
```
- Served via **Uvicorn**: `uvicorn predict:app --host 0.0.0.0 --port 9696`.

---

## Summary (Module 5)
Module 5 bridges the gap between machine learning experimentation and real-world software engineering. We learned how to serialize trained models and feature vectorizers using `pickle` and encapsulate training routines in standalone scripts (`train.py`). We converted inference pipelines into web services using `Flask` and `FastAPI`, addressed production serving needs with WSGI/ASGI servers (`gunicorn`, `waitress`, `uvicorn`), and managed deterministic dependencies using `Pipenv` and `uv`. Finally, we packaged the entire application stack into immutable `Docker` containers and deployed them to cloud platforms (`AWS Elastic Beanstalk`, `Fly.io`), establishing a robust, scalable, and reproducible ML serving pipeline.

---

## Glossary
- **Model Deployment**: The process of integrating a machine learning model into an active production environment where it can accept input data and output predictions.
- **Serialization**: The process of converting an in-memory Python object into a persistent byte stream (e.g., via `pickle.dump`).
- **Deserialization**: The process of reconstructing an in-memory Python object from a serialized byte stream (e.g., via `pickle.load`).
- **Web Service**: A software system designed to support interoperable machine-to-machine interaction over a network via standard protocols like HTTP.
- **HTTP Methods (Verbs)**: Actions defined in the HTTP protocol; most notably `GET` (retrieving information) and `POST` (submitting data payloads).
- **Endpoint / Route**: A specific URL path where a web service can be accessed to perform dedicated operations.
- **JSON (JavaScript Object Notation)**: A lightweight, human-readable data interchange format standard for web APIs.
- **WSGI (Web Server Gateway Interface)**: A standardized convention for web servers to forward requests to Python web applications (e.g., Gunicorn, Waitress).
- **ASGI (Asynchronous Server Gateway Interface)**: The modern asynchronous successor to WSGI, supporting concurrent async execution (e.g., Uvicorn).
- **Virtual Environment**: An isolated directory tree containing a dedicated Python installation and set of libraries, preventing dependency collisions across projects.
- **Pipfile & Pipfile.lock**: Configuration files used by Pipenv; `Pipfile` lists project dependencies, while `Pipfile.lock` specifies exact versions and cryptographic hashes for reproducibility.
- **uv**: An extremely fast, modern Python package and project manager written in Rust by Astral.
- **Docker**: A containerization platform that packages applications and their underlying operating system environments into portable, isolated containers.
- **Docker Image**: A read-only blueprint containing all instructions, dependencies, and code needed to instantiate a Docker container.
- **Docker Container**: A lightweight, standalone, executable runtime instance of a Docker image.
- **Port Mapping (`-p host:container`)**: Publishing a port inside a Docker container to a specific port on the host machine.
- **AWS Elastic Beanstalk (EB)**: An AWS Platform-as-a-Service (PaaS) that automates deployment, provisioning, load balancing, and scaling of containerized applications.
- **Pydantic**: A data validation and settings management library using Python type hints to enforce schema integrity.
