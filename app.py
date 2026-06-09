import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
import chromadb
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import torchvision.transforms as transforms

app = FastAPI(
    title="NIH ChestX-ray14 CV-RAG Diagnostic Support Engine",
    description="Assistive Vision-Language Pipeline for Thoracic Screening Support."
)

# 1. Pipeline Static Configuration Variables
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class CustomChestCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(CustomChestCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 2. Asset Initialization 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomChestCNN(num_classes=14).to(device)
model.load_state_dict(torch.load("custom_baseline_cnn.pth", map_location=device))
model.eval()

# --- SMART AUTO-INITIALIZATION FOR CHROMADB ---
chroma_client = chromadb.PersistentClient(path="./nih_medical_rag_db")

# Use get_or_create_collection so it never crashes if empty
rag_collection = chroma_client.get_or_create_collection(name="thoracic_guidelines")

# Standard medical reference text to populate the local vector engine
medical_textbook_data = [
    "Atelectasis context: Characterized by a localized volume loss within the lung structures. Often overlaps visually with infiltration signs. Radiologists must review for associated shift or diaphragmatic elevation.",
    "Cardiomegaly context: Defined by a cardiothoracic ratio exceeding 50 percent on standard PA projection views. Note that portable anteroposterior views can artificially exaggerate heart dimensions.",
    "Effusion context: Represents fluid accumulation in the pleural space, presenting as costophrenic angle blunting. Small fluid volumes are easily obscured if the patient is imaged in a supine position.",
    "Infiltration context: Indicates an abnormal substance like fluid, pus, or cells pooling inside the parenchyma. Visually presents as ill-defined opacity or cloudiness across the lung fields."
]
document_ids = [f"doc_{i}" for i in range(len(medical_textbook_data))]

# Safely seed documents if the collection is brand new
if rag_collection.count() == 0:
    rag_collection.add(documents=medical_textbook_data, ids=document_ids)
# -----------------------------------------------


norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std)
])

@app.post("/analyze-chest-xray/")
async def analyze_chest_xray(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
        
    try:
        # Read incoming binary stream file components safely into PIL
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        tensor = inference_transform(image).unsqueeze(0).to(device)
        
        # Compute vision probabilities
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
        # Package prediction dictionaries mapping label keys to probability floats
        predictions_dictionary = {DISEASE_LABELS[i]: float(probabilities[i]) for i in range(len(DISEASE_LABELS))}
        
        # Route the top predicted disease path to our RAG textbook system
        primary_finding = max(predictions_dictionary, key=predictions_dictionary.get)
        primary_confidence = predictions_dictionary[primary_finding] * 100
        
        # Query ChromaDB collection index records
        search_results = rag_collection.query(query_texts=[primary_finding], n_results=1)
        retrieved_snippet = search_results['documents'][0] if search_results['documents'] else ["General observations guidelines."]
        
        # Synthesize structured clinical response body log entries
        clinical_report = (
            f"=== ASSISTIVE RADIOLOGY INTERPRETATION REPORT ===\n"
            f"Primary Detected Finding: {primary_finding}\n"
            f"Confidence Score: {primary_confidence:.2f}%\n\n"
            f"[Source Citation 1]: \"{retrieved_snippet}\"\n\n"
            f"Analysis suggests visual manifestations of {primary_finding}. Cross-examine with criteria in [Source Citation 1].\n\n"
            f"⚠️ DISCLAIMER: Assistive system software layer. Not a definitive medical diagnosis."
        )
        
        return {
            "status": "Success",
            "detected_pathologies": predictions_dictionary,
            "grounded_interpretation": clinical_report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal pipeline processing failure: {str(e)}")
