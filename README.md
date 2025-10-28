web app link:  https://smartcalai.streamlit.app/

## Project Summary

**SmartCalc AI** is a machine learning-powered calculator that predicts electricity bill, units consumed, or connected load based on any two user inputs. It uses cleaned residential electricity data, engineered features, and trained XGBoost models to deliver fast, accurate predictions.

The pipeline includes:
-  Data cleaning and residential filtering  
- Feature engineering (trend, efficiency, averages)  
- Outlier removal using IQR  
- Model training with GridSearchCV and StandardScaler  
- A smart prediction function that fills missing features using mean values  
-  Saved models and scalers for deployment  
-  A Streamlit frontend with clean UI and instant results

Users enter any two of:  
🔌 Load (kW), ⚡ Units (kWh), 💸 Bill (₹) — and SmartCalc AI predicts the third.

The app is deployed on Streamlit Cloud, and the full codebase includes:
- Modular folder structure  
- Smart prediction logic in `predictor.py`  
- Feature fallback via `feature_means.csv`  
- Clean UI in `app.py`  
- README, report, and presentation for documentation

This project reflects real-world ML deployment, modular design, and clean UI/UX — built for demo impact and recruiter visibility.

---


