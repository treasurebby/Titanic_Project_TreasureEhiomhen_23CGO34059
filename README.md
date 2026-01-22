# 🚢 Titanic Survival Prediction System

A machine learning web application that predicts passenger survival on the Titanic using Random Forest Classifier.

## 📋 Project Overview

This project implements a complete machine learning pipeline for predicting Titanic passenger survival, including:
- Data preprocessing and feature engineering
- Random Forest classification model
- Flask web application with modern UI
- Model persistence using Joblib

## 🎯 Features Selected

The model uses the following 5 features:
1. **Pclass** - Passenger Class (1st, 2nd, 3rd)
2. **Sex** - Gender (Male/Female)
3. **Age** - Passenger Age
4. **Fare** - Ticket Fare
5. **Embarked** - Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

**Target Variable:** Survived (0 = No, 1 = Yes)

## 🛠️ Technology Stack

- **Backend:** Flask (Python)
- **Machine Learning:** scikit-learn
- **Model Persistence:** Joblib
- **Frontend:** HTML, CSS
- **Deployment:** Render.com 

## 📁 Project Structure

```
Titanic_Project_TreasureEhiomhen_23CGO34059/
│
├── app.py                              # Flask web application
├── requirements.txt                    # Python dependencies
├── Titanic_hosted_webGUI_link.txt     # Submission details
├── README.md                          # Project documentation
│
├── model/
│   ├── model_building.py              # Model training script
│   ├── titanic_survival_model.pkl     # Trained model
│   ├── scaler.pkl                     # Feature scaler
│   ├── le_sex.pkl                     # Sex label encoder
│   └── le_embarked.pkl                # Embarked label encoder
│
├── static/
│   └── style.css                      # CSS styling
│
└── templates/
    └── index.html                     # HTML template
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone (https://github.com/treasurebby/Titanic_Project_TreasureEhiomhen_23CGO34059)
cd Titanic_Project_TreasureEhiomhen_23CGO34059
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model
```bash
python model/model_building.py
```

This will:
- Load the Titanic dataset
- Preprocess the data
- Train the Random Forest model
- Save the model and preprocessors to the `model/` directory
- Display model evaluation metrics

### Step 4: Run the Web Application
```bash
python app.py
```

The application will start at `http://localhost:5000`

## 📊 Model Performance

The Random Forest Classifier achieves:
- **Accuracy:** ~80-85%
- **Features Used:** 5 (Pclass, Sex, Age, Fare, Embarked)
- **Algorithm:** Random Forest with 100 estimators

### Classification Report Example:
```
              precision    recall  f1-score   support

Did Not Survive     0.84      0.87      0.85       105
      Survived      0.80      0.76      0.78        74

      accuracy                          0.82       179
```


## 💻 Usage

1. Open the web application in your browser
2. Enter passenger details:
   - Select Passenger Class (1st, 2nd, or 3rd)
   - Select Gender
   - Enter Age
   - Enter Fare amount
   - Select Port of Embarkation
3. Click "Predict Survival"
4. View the prediction result with probability percentages

## 🧪 API Endpoint

The application also provides a JSON API endpoint:

```bash
POST /api/predict
Content-Type: application/json

{
    "pclass": 3,
    "sex": "male",
    "age": 22,
    "fare": 7.25,
    "embarked": "S"
}
```

Response:
```json
{
    "success": true,
    "prediction": 0,
    "result": "Did Not Survive",
    "probabilities": {
        "survived": 0.15,
        "not_survived": 0.85
    },
    "confidence": 0.85
}
```

## 📝 Model Building Process

1. **Data Loading:** Load Titanic dataset from CSV or seaborn
2. **Data Preprocessing:**
   - Handle missing values (Age, Fare, Embarked)
   - Encode categorical variables (Sex, Embarked)
   - Scale numerical features
3. **Model Training:** Train Random Forest with optimal hyperparameters
4. **Model Evaluation:** Generate classification report and confusion matrix
5. **Model Persistence:** Save model and preprocessors using Joblib

## 🔧 Customization

You can modify the model by editing `model/model_building.py`:
- Change algorithm (Logistic Regression, SVM, KNN)
- Adjust hyperparameters
- Add/remove features
- Change train-test split ratio

## 📚 References

- [Titanic Dataset - Kaggle](https://www.kaggle.com/c/titanic)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)

## 👤 Author

**Ehiomhen Treasure**  
Matric Number: 23CG034059
Date: January 2026

## 📄 License

This project is created for academic purposes.

---

**Note:** Make sure to run `model_building.py` before running `app.py` to generate the necessary model files.
