# # Predicting Antibiotic Resistance Using Genomic Data and Machine Learning

## 👨‍🔬 Team Members
- **Karthik R S**  
- **Mohamed Hani**
- **Adith Anish** 

---

## 🧠 Project Overview

Antibiotic resistance is a critical global health threat. This project focuses on predicting antibiotic resistance in *Neisseria gonorrhoeae* using genomic data and machine learning models. Traditional culture-based diagnostic methods are slow and limited; our approach aims to improve the speed and accuracy of resistance detection.

---

## 🧬 Motivation

- *Neisseria gonorrhoeae*, responsible for gonorrhea, has developed resistance to nearly all major antibiotics.
- WHO classifies drug-resistant gonorrhea as a **high-priority global threat**.
- Genomic sequencing reveals DNA-based resistance markers.
- Machine Learning enables **rapid, scalable, and cost-effective** prediction.

---

## 📂 Dataset Description

Datasets used:
- `azm_sr_gwas_filtered_unitigs.csv`: Azithromycin (AZM) 
- `cip_sr_gwas_filtered_unitigs.csv`: Ciprofloxacin (CIP) 
- `cfx_sr_gwas_filtered_unitigs.csv`: Cefixime (CFX) 
- `metadata.csv`: Sample metadata with resistance phenotype labels

### Format:
- **Rows**: Each row = a bacterial isolate
- **Columns**: First column = Sample ID; others = unitigs (binary: 1 = present, 0 = absent)

---

## ⚙️ Methods & Techniques

### ML Algorithms:
- XGBoost
- Decision Tree
- Random Forest
- SVM

### Techniques Used:
- Preprocessing
- Cross-validation
- Hyperparameter tuning (e.g., GridSearchCV)
- Evaluation metrics (Accuracy, Balanced Accuracy, Confusion Matrix)
- Data Visualization (e.g., Seaborn/Matplotlib plots)

---

## 📈 Results

Models were evaluated separately for AZM, CIP, and CFX datasets. Each was analyzed using confusion matrices and classification reports. The best-performing models varied by antibiotic, with **XGBoost** and **Random Forest** showing strong performance. The results are available in folder "Results"

---

## 📖 References

1. [[Kaggle Dataset – Neisseria gonorrhoeae Unitigs](https://www.kaggle.com/datasets/nwheeler443/gono-unitigs)  ]
2. Sakagianni et al., "Using Machine Learning to Predict Antimicrobial Resistance," *Antibiotics*, 2023.  
3. Jaillard et al., "A fast and agnostic method for bacterial genome-wide association studies," *PLOS Genetics*, 2018.

---

## 📌 Goal

Enable accurate and fast antibiotic resistance prediction from genome data, assisting healthcare and research institutions in combating drug-resistant gonorrhea.

