
# Health Insurance Premium Prediction App

This Streamlit web application predicts the **expected health insurance premium** based on a user’s personal, health, and lifestyle details. It helps users estimate their annual premium by leveraging a machine learning model trained on real-world data.

---

## Features

- Simple and intuitive UI using Streamlit
- Real-time prediction of health insurance premium
- Input form includes:
  - Age, income, dependents
  - Genetic risk, BMI, medical history
  - Employment status, region, marital status
  - Smoking habits and more
- Backend model trained with regression algorithm
- One-hot and label encoding used during preprocessing
- Hosted on [Streamlit Cloud](https://moksh-health-insurance-premium-prediction.streamlit.app/)

---

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Pandas, NumPy, Scikit-learn
- **Deployment**: Streamlit Cloud
- **Model**: Linear Regression & XGB Regressor

---

## Machine Learning Details

- Used `pd.get_dummies()` for one-hot encoding nominal features
- Data was preprocessed and matched exactly with training format
- Model serialized with `joblib`
- Scaler and encoders handled carefully to maintain consistency during prediction

---

##  How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/itsmoksh/health-insurance-premium-prediction.git
   cd health-insurance-premium-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## Live App

Check out the **live demo**:  
[Streamlit Cloud Link](#) *(https://moksh-health-insurance-premium-prediction.streamlit.app/)*

---

## Screenshots
## Screenshots

![Age more than 25](assets/image1.jpg)  
![Age less than equal to 25](assets/image2.jpg)

---

##  Author

**Moksh Jain**  
Aspiring Data Scientist | Python & ML Enthusiast  
[LinkedIn](https://www.linkedin.com/in/itsmoksh/) • [GitHub](https://github.com/itsmoksh)
