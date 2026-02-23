# Federated Learning Based Crop Disease Detection

This project implements a Federated Learning system for detecting crop diseases using deep learning.

Instead of sending all data to one central server, multiple clients train the model locally on their own datasets. Only the model weights are shared with the central server. The server aggregates these weights to improve the global model.

This helps in maintaining data privacy while still achieving good accuracy.

---

## Problem Statement

Crop diseases reduce agricultural productivity. Traditional machine learning approaches require centralized data collection, which may not be practical or privacy-friendly.

Federated Learning solves this problem by allowing decentralized model training.

---

## Features

- Client-server federated architecture
- Local training using CNN (MobileNetV2)
- Federated Averaging for weight aggregation
- Global model evaluation
- Web interface for disease prediction

---

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- Flask
- NumPy
- Pandas

---

## Project Structure

- app.py → Main web application
- training/local_train.py → Local client training
- training/federated_avg.py → Federated averaging logic
- training/evaluate_global.py → Global model evaluation
- models/ → Saved global model
- templates/ → HTML frontend

---

## How It Works

1. Server initializes a global model.
2. Clients receive the global model.
3. Each client trains the model locally.
4. Clients send updated weights to the server.
5. Server performs federated averaging.
6. Global model gets updated.

This process repeats for multiple rounds.

---

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   python app.py

3. Open in browser:
   http://127.0.0.1:5000

---

## Note

Dataset is not uploaded due to large size.

## Why Federated Learning?

Federated learning improves privacy because raw data never leaves the client devices. Only model updates are shared.