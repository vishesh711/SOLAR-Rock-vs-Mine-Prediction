# SOLAR-Rock-vs-Mine-Prediction

## Overview
This project aims to classify objects detected by a submarine sonar as either a Mine (M) or a Rock (R). Using a supervised learning approach, we employ a Logistic Regression model to handle this binary classification problem.

## Workflow
1. **Data Collection and Preprocessing**
2. **Train-Test Split**
3. **Model Training**
4. **Model Evaluation**
5. **Prediction System**

## Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Data Collection and Preprocessing
1. **Loading the Dataset**
    ```python
    sonar_data = pd.read_csv('/content/sonar data.csv', header=None)
    sonar_data.head()
    ```
2. **Exploring the Data**
    ```python
    sonar_data.shape
    sonar_data.describe()
    sonar_data[60].value_counts()
    ```

3. **Separating Data and Labels**
    ```python
    X = sonar_data.drop(columns=60, axis=1)
    Y = sonar_data[60]
    print(X)
    print(Y)
    ```

## Training and Test Data
1. **Splitting the Data**
    ```python
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
    print(X.shape, X_train.shape, X_test.shape)
    print(X_train)
    print(Y_train)
    ```

## Model Training
1. **Training the Logistic Regression Model**
    ```python
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    ```

## Model Evaluation
1. **Accuracy on Training Data**
    ```python
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy on training data : ', training_data_accuracy)
    ```

2. **Accuracy on Test Data**
    ```python
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy on test data : ', test_data_accuracy)
    ```

## Predictive System
1. **Making Predictions**
    ```python
    input_data = (0.0264,0.0071,0.0342,0.0793,0.1043,0.0783,0.1417,0.1176,0.0453,0.0945,0.1132,0.0840,0.0717,0.1968,0.2633,0.0191,0.5050,0.6711,0.7922,0.8381,0.8759,0.9422,1.0000,0.9931,0.9575,0.8647,0.7215,0.5801,0.4964,0.4886,0.4079,0.2443,0.1768,0.2472,0.3518,0.3762,0.2909,0.2311,0.3168,0.3554,0.3741,0.4443,0.3261,0.1963,0.0864,0.1688,0.1991,0.1217,0.0628,0.0323,0.0253,0.0214,0.0262,0.0177,0.0037,0.0068,0.0121,0.0077,0.0078,0.0066)
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = model.predict(input_data_reshaped)
    
    print(prediction)
    if (prediction[0]=='R'):
      print('Rock')
    else:
      print('Mine')
    ```

## Results
- **Accuracy on Training Data:** 83.42%
- **Accuracy on Test Data:** 76.19%
- **Sample Prediction:** Mine

## Conclusion
This project successfully classifies sonar data into Mines and Rocks using a Logistic Regression model. Future improvements can include exploring other models, performing hyperparameter tuning, and implementing cross-validation to enhance the model's performance.

## License
This project is licensed under the MIT License.

## Contact
For any queries, feel free to contact Vishesh Kumar at mail4vishesh@gmail.com

---
