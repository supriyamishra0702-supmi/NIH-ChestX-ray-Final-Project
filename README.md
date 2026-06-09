# NIH-ChestX-ray-Final-Project
An assistive clinical decision support pipeline that combines Deep Learning computer vision with Retrieval-Augmented Generation (RAG) to screen frontal chest radiographs for 14 distinct thoracic conditions.
# NIH ChestX-ray14: Multi-Label Thoracic Disease Classification with CNN + RAG-Grounded LLM Interpretation Layer

An assistive clinical decision support pipeline that combines Deep Learning computer vision with Retrieval-Augmented Generation (RAG) to screen frontal chest radiographs for 14 distinct thoracic conditions.

## 🚀 Key Project Achievements
* **Leakage-Safe Data Split**: Structured strict patient-wise partitioning to isolate patient anatomical biases completely between train and test boundaries.
* **Imbalance Mitigation**: Integrated a dynamic positive class-weight loss matrix to optimize deep learning paths for rare thoracic diseases.
* **Factual Grounding (RAG)**: Built a local `ChromaDB` vector index to feed verified diagnostic notes directly into an uncertainty-aware report pipeline.

## 📊 Quantitative Evaluation Metrics Table
The pipeline achieved high ranking accuracy across highly imbalanced validation distributions:

| Pathology Class | Validation AUROC Score | Validation PR-AUC Score |
| :--- | :---: | :---: |
| **Hernia** | 0.8853 | 0.0146 |
| **Edema** | 0.8567 | 0.2330 |
| **Emphysema** | 0.8318 | 0.1104 |
| **Consolidation** | 0.8178 | 0.0802 |
| **Cardiomegaly** | 0.8127 | 0.1529 |
| **Pneumothorax** | 0.8025 | 0.1378 |
| **Pleural_Thickening**| 0.7767 | 0.0990 |
| **Effusion** | 0.7674 | 0.2219 |
| **Atelectasis** | 0.7507 | 0.2439 |
| **Fibrosis** | 0.6948 | 0.0328 |
| **Infiltration** | 0.6612 | 0.3152 |
| **Mass** | 0.6577 | 0.1454 |
| **Pneumonia** | 0.6397 | 0.0228 |
| **Nodule** | 0.5592 | 0.0598 |
| **GLOBAL MEAN SUMMARY** | **0.7510** | **0.1336** |

---

## 📝 Example Output Report Log
When the system analyzes an image displaying anomalies, it references factual database notes to generate a structured report:

```text
=== ASSISTIVE RADIOLOGY INTERPRETATION REPORT ===
Target Finding Analysed: Infiltration
Classifier Confidence Level: 76.54%

Clinical Reference Grounding:
[Source Citation 1]: "Infiltration context: Indicates an abnormal substance like fluid, pus, or cells pooling inside the parenchyma. Visually presents as ill-defined opacity or cloudiness across the lung fields."

⚠️ CRITICAL SYSTEM DISCLAIMER:
This system is an assistive decision support software layer designed for prioritization and clinical screening support. It does NOT provide a definitive medical diagnosis. Final image interpretation and diagnostic confirmation must be performed exclusively by a qualified radiologist.
```
## Here is the screenshot of the RAG portal:
<img width="927" height="436" alt="NIH_API Deployment layer" src="https://github.com/user-attachments/assets/5d10bc48-9a02-4c3c-a414-86d95a6031d2" />


## Below is the screenshot of the response after uploading a chest xray image:
<img width="890" height="468" alt="portal_demo jpg" src="https://github.com/user-attachments/assets/3105f1d3-e267-4a5c-adf0-1e4c1959053f" />

