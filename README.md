# Skin-Disease-Detection-using-CNN

## Installation

### Install Requirements
pip install -r requirements.txt

### Run
python -m uvicorn main:app

This would start the ASGI server at http://127.0.0.1:8000.

## Using Front-End

 - After starting the server open http://127.0.0.1:8000 in your browser.
 - Upload the image that you want to examine for diseases.
 - Click Diagnose
 - Detailed log of Disease predicted by the neural networks and its symptoms and duration will pop up

## Using API Directly

Request URL:
```
 http://127.0.0.1:8000/predict

```

Using Swagger UI:
 - After starting the server open http://127.0.0.1:8000/docs in your browser to open the swagger UI for APIs.
 - Scroll down to Post /predict
 - Click Try it Out
 - Chose file to upload for classification 
 - Click Execute
 - Under Responses section , a detailed log of disease, summary ,symptoms and duration will be displayed in json format as the response 
