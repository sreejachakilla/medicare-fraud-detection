Project Overview
Healthcare fraud results in billions of dollars in financial losses each year, affecting the sustainability and integrity of healthcare systems. Traditional rule-based and statistical fraud detection systems struggle to adapt to evolving fraud patterns and often produce high false positives.
This project proposes a Hybrid Deep Learning and Machine Learning Framework that integrates:
Convolutional Neural Networks (CNN)
Transformer Architecture
XGBoost Classifier
SHAP Explainable AI
The system is deployed as a Flask-based web application that enables real-time fraud prediction and interpretability for Medicare claims.


Motivation
Healthcare fraud is continuously evolving, making traditional detection systems ineffective. Manual audits are time-consuming and prone to bias. There is a strong need for:
Intelligent automated detection
High predictive accuracy
Transparency in decision-making
Real-time usability
This project aims to address both performance and interpretability challenges in fraud detection.

Objectives
Develop a hybrid fraud detection model combining CNN, Transformer, and XGBoost.
Improve fraud detection accuracy while reducing false positives.
Handle imbalanced healthcare claim datasets effectively.
Provide feature-level interpretability using SHAP.
Deploy the system as a user-friendly web application.

Proposed System
The proposed system integrates deep learning, ensemble learning, and explainable AI into a unified architecture:
Data Preprocessing (Cleaning, Encoding, Scaling, Feature Alignment)
Feature Extraction using CNN (Local Dependencies)
Context Modeling using Transformer (Global Relationships)
Feature Fusion (CNN + Transformer Outputs)
Final Classification using XGBoost
SHAP-based Explainability
Real-time Deployment using Flask
The hybrid fusion model significantly outperforms traditional machine learning models.

Model Architecture
🔹 CNN
Captures localized patterns in healthcare claims such as abnormal billing behavior and irregular claim frequencies.
🔹 Transformer
Uses self-attention mechanisms to capture global contextual relationships across claims data.
🔹 XGBoost
Performs robust classification on structured and fused feature representations with high accuracy.
🔹 Hybrid Model
CNN and Transformer feature outputs are concatenated and passed into XGBoost for final fraud classification, leveraging both local and global insights.

Explainable AI (SHAP)
To ensure transparency and trust:
SHAP (SHapley Additive exPlanations) is integrated.
Provides feature-level importance scores.
Visualizes attributes influencing fraud detection.
Enhances accountability and interpretability in healthcare decision-making.
This ensures the system is not treated as a black-box model.

Tech Stack
Programming Language:
Python 3.8+
Machine Learning / Deep Learning:
TensorFlow / Keras
XGBoost
Scikit-learn
SHAP

Web Framework:
Flask
HTML, CSS, Bootstrap

Database:
MySQL

Libraries:
Pandas
NumPy
Joblib
Matplotlib

How to Run
git clone https://github.com/sreejachakilla/medicare-fraud-detection.git
cd medicare-fraud-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

Open browser:
http://127.0.0.1:5000/

Results
Hybrid model outperformed individual models (CNN, Transformer, RF, DT).
Improved fraud detection accuracy.
Reduced false positives.
Provided interpretable feature-level explanations using SHAP.
Successfully deployed as a real-time web application.

Future Scope
Integration of federated learning for privacy preservation.
Real-time streaming fraud detection.
Cloud deployment (AWS/Azure/GCP).
Advanced explainability techniques (LIME, Counterfactual explanations).
Integration with enterprise healthcare systems.

