import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)


def random_value_imputation(df, feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample
    return df


def impute_mode(df, feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)
    return df


def impute_mean(df, feature):
    mean = df[feature].mean()
    df[feature] = df[feature].fillna(mean)
    return df


# This function reads the data from csv file and returns the cleaned after pre-processing
def get_heart_disease_data():
    # Read the data from csv file
    data = pd.read_csv('data/heart_disease_data.csv')

    return data


# This function reads the data from csv file and returns the cleaned after pre-processing
def get_kidney_disease_data():
    # Read the data from csv file
    data = pd.read_csv('data/kidney_disease.csv')

    # Drop ID columns
    data = data.drop(columns=['id'])

    # Replace both '\t' and '\t?' characters with an empty string across the entire DataFrame
    data = data.replace(r'\t\??', '', regex=True)

    # Replace the values in classification column. 1 reflects kidney disease and 0 reflects no kidney disease
    data['classification'] = data['classification'].replace('ckd', 1)
    data['classification'] = data['classification'].replace('notckd', 0)

    # Remove extra spaces
    data['dm'] = data['dm'].str.strip()

    # Convert the columns to numeric
    col_object_to_numeric = ['pcv', 'wc', 'rc']
    for col in col_object_to_numeric:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Extracting categorical and numerical columns
    cat_cols = [col for col in data.columns if data[col].dtype == 'O']
    num_cols = [col for col in data.columns if data[col].dtype != 'O']

    # Imputing Numerical Columns
    # We will impute 'age' and 'bp' with mean value and rest of the numerical columns with random sampling
    data = impute_mean(data, 'age')
    data = impute_mean(data, 'bp')
    for col in num_cols:
        data = random_value_imputation(data, col)

    # Imputing Categorical Columns
    # We will impute 'rbc' and 'pc' with mode value and rest of the categorical columns with random sampling
    data = random_value_imputation(data, 'rbc')
    data = random_value_imputation(data, 'pc')
    for col in cat_cols:
        data = impute_mode(data, col)

    # Converting Categorical features using Label Encoder
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])

    return data


# This function reads the data from csv file and returns the cleaned after pre-processing
def get_liver_disease_data():
    # Read the data from csv file
    data = pd.read_csv('data/liver_data.csv')

    data['Dataset'] = data['Dataset'].replace(2, 0)

    # Impute the column with missing values with mean value
    mean_data = data['Albumin_and_Globulin_Ratio'].mean()
    data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(mean_data)

    # Convert categorical features using Label Encoder
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])

    return data


# This function reads the data from csv file and returns the cleaned after pre-processing
def get_diabetes_disease_data():
    # Read the data from csv file
    df = pd.read_csv('data/diabetes_disease_data.csv')
    # Some of the values in Glucose, BloodPressure, SkinThickness, Insulin, BMI are 0 which reflects either missing
    # data or measurement error as these values
    # can't be 0. To maintain the data integrity, replace the 0 values with mean value.

    # Replace 0 values in Glucose column with mean value of the Glucose
    mean_bp = df[df['Glucose'] != 0]['Glucose'].mean()
    df['Glucose'] = df['Glucose'].replace(0, mean_bp)

    # Replace 0 values in Blood Pressure column with mean value of the blood pressure
    mean_bp = df[df['BloodPressure'] != 0]['BloodPressure'].mean()
    df['BloodPressure'] = df['BloodPressure'].replace(0, mean_bp)

    # Replace 0 values in Skin Thickness column with mean value of the skin thickness
    mean_bp = df[df['SkinThickness'] != 0]['SkinThickness'].mean()
    df['SkinThickness'] = df['SkinThickness'].replace(0, mean_bp)

    # Replace 0 values in Insulin column with mean value of the Insulin
    mean_bp = df[df['Insulin'] != 0]['Insulin'].mean()
    df['Insulin'] = df['Insulin'].replace(0, mean_bp)

    # Replace 0 values in BMI column with mean value of the BMI
    mean_bp = df[df['BMI'] != 0]['BMI'].mean()
    df['BMI'] = df['BMI'].replace(0, mean_bp)

    return df


# This function reads the data from csv file and return the cleaned data after pre-processing
def get_parkinson_disease_data():
    # Read the data from csv file
    data = pd.read_csv('data/parkinsons_disease_data.csv')
    data = data.drop(['name'], axis=1)

    return data


# This function reads the data from csv file and return the cleaned data after pre-processing
def get_wine_quality_data():
    # Read the data from csv file
    data = pd.read_csv('data/wine_data.csv')
    data['quality'] = data['quality'].apply(lambda x: 1 if x > 5 else 0)

    return data


# This function reads the data from csv file and return the cleaned data after pre-processing
def get_algerian_forest_fire_data():
    # Read the data from csv file
    data = pd.read_csv('data/algerian_forest_fire_data.csv')

    # Drop month day and year
    data = data.drop(['day', 'month', 'year'], axis=1)

    # Encoding of categories in classes
    data['Classes'] = np.where(data['Classes'].str.contains('not fire'), 0, 1)

    return data


# This function reads the data from csv file and return the cleaned data after pre-processing
def get_titanic_survival_data():
    # Read the data from csv file
    data = pd.read_csv('data/titanic_dataset.csv')

    # Fill the missing values in Age and Embarked column with mean and mode value respectively
    data = impute_mean(data, 'Age')
    data = impute_mode(data, 'Embarked')

    # Drop the Cabin column as it has too many null values
    data = data.drop('Cabin', axis=1)

    # Converting categorical features to numeric
    embarked = pd.get_dummies(data['Embarked'], drop_first=True)
    sex = pd.get_dummies(data['Sex'], drop_first=True)
    data.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)  # Drop un-necessary columns
    data = pd.concat([data, sex, embarked], axis=1)  # Concatenate the one-hot encoded columns with data
    data = data.rename(columns={'male': 'Sex', 'Q': 'Embarked_Q',
                                'S': 'Embarked_S'})  # Rename to columns to meaningful names

    return data


# This function reads the data from csv file and return the cleaned data after pre-processing
def get_penguins_data():
    # Read the data from csv file
    data = pd.read_csv('data/penguins_cleaned.csv')

    # Converting categorical features to numeric
    island = pd.get_dummies(data['island'], drop_first=True)
    sex = pd.get_dummies(data['sex'], drop_first=True)
    data.drop(['sex', 'island'], axis=1, inplace=True)  # Drop un-necessary columns
    data = pd.concat([data, sex, island], axis=1)  # Concatenate the one-hot encoded columns with data
    data = data.rename(columns={'male': 'sex', 'Dream': 'island_dream',
                                'Torgersen': 'island_torgersen'})  # Rename to columns to meaningful names

    # Encoding of Species
    species_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

    def species_encoder(val):
        return species_mapper[val]

    data['species'] = data['species'].apply(species_encoder)
    return data


# This function reads the data from csv file and return the cleaned data after pre-processing
def get_iris_data():
    # Read the data from csv file
    data = pd.read_csv('data/iris.csv')

    # Encoding of Flower Species
    species_mapper = {'setosa': 0, 'versicolor': 1, 'virginica': 2}

    def species_encoder(val):
        return species_mapper[val]

    data['Species'] = data['Species'].apply(species_encoder)
    return data


# Function to train the machine learning model
def train_model(model_name, dataset, label):

    # Setting the default value of Model Parameters. Parameters will be updated based on dataset selection
    svm_params = {'C': 1, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True}
    lr_params = {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs'}
    knn_params = {'algorithm': 'auto', 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}

    if dataset == "Heart":
        # Get the data
        data = get_heart_disease_data()

        # Get the dependent and independent features
        X = data.drop(['target'], axis=1)
        y = data['target']

        # Model Parameters
        svm_params = {'C': 50, 'gamma': 0.01, 'kernel': 'linear', 'probability': True}
        lr_params = {'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 8, 'p': 2, 'weights': 'distance'}

    elif dataset == "Diabetes":
        # Get the data
        data = get_diabetes_disease_data()

        # Get the dependent and independent features
        X = data.drop(['Outcome'], axis=1)
        y = data['Outcome']

        # Model Parameters
        svm_params = {'C': 0.1, 'gamma': 1, 'kernel': 'linear', 'probability': True}
        lr_params = {'C': 0.1, 'penalty': 'l1', 'solver': 'saga'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}

    elif dataset == "Parkinson":
        # Get the data
        data = get_parkinson_disease_data()

        # Get the dependent and independent features
        X = data.drop(['status'], axis=1)
        y = data['status']

        # Model Parameters
        svm_params = {'C': 10, 'gamma': 0.1, 'kernel': 'rbf', 'probability': True}
        lr_params = {'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 1, 'p': 2, 'weights': 'uniform'}

    elif dataset == "Liver":
        # Get the data
        data = get_liver_disease_data()

        # Get the dependent and independent features
        X = data.drop(['Dataset'], axis=1)
        y = data['Dataset']

        # Model Parameters
        svm_params = {'C': 1, 'gamma': 0.01, 'kernel': 'rbf', 'probability': True}
        lr_params = {'C': 100, 'penalty': 'l2', 'solver': 'saga'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 9, 'p': 1, 'weights': 'distance'}

    elif dataset == "Kidney":
        # Get the data
        data = get_kidney_disease_data()

        # Get the dependent and independent features
        X = data.drop(['classification'], axis=1)
        y = data['classification']

        # Model Parameters
        svm_params = {'C': 1, 'gamma': 0.01, 'kernel': 'rbf', 'probability': True}
        lr_params = {'C': 100, 'penalty': 'l2', 'solver': 'saga'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 1, 'p': 1, 'weights': 'uniform'}

    elif dataset == "Wine":
        # Get the data
        data = get_wine_quality_data()

        # Get the dependent and independent features
        X = data.drop(['quality'], axis=1)
        y = data['quality']

        # Model Parameters
        svm_params = {'C': 40, 'gamma': 0.1, 'kernel': 'rbf', 'probability': True}
        lr_params = {'C': 0.1, 'penalty': 'l1', 'solver': 'saga'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 10, 'p': 1, 'weights': 'distance'}

    elif dataset == "Algerian Forest Fire":
        # Get the data
        data = get_algerian_forest_fire_data()

        # Get the dependent and independent features
        X = data.drop(['Classes'], axis=1)
        y = data['Classes']

        # Model Parameters
        svm_params = {'C': 30, 'gamma': 1, 'kernel': 'linear', 'probability': True}
        lr_params = {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 7, 'p': 1, 'weights': 'uniform'}

    elif dataset == "Titanic":
        # Get the data
        data = get_titanic_survival_data()

        # Get the dependent and independent features
        X = data.drop(['Survived'], axis=1)
        y = data['Survived']

        # Model Parameters
        svm_params = {'C': 10, 'gamma': 0.1, 'kernel': 'rbf', 'probability': True}
        lr_params = {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 10, 'p': 1, 'weights': 'uniform'}

    elif dataset == "Penguin":
        # Get the data
        data = get_penguins_data()

        # Get the dependent and independent features
        X = data.drop(['species'], axis=1)
        y = data['species']

        # Model Parameters
        svm_params = {'C': 1, 'gamma': 0.1, 'kernel': 'rbf', 'probability': True}
        lr_params = {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 2, 'p': 1, 'weights': 'uniform'}

    elif dataset == "Iris":
        # Get the data
        data = get_iris_data()

        # Get the dependent and independent features
        X = data.drop(['Species'], axis=1)
        y = data['Species']

        # Model Parameters
        svm_params = {'C': 100, 'gamma': 0.001, 'kernel': 'linear', 'probability': True}
        lr_params = {'C': 100, 'penalty': 'l1', 'solver': 'saga'}
        knn_params = {'algorithm': 'auto', 'n_neighbors': 7, 'p': 1, 'weights': 'distance'}

    # Split the data to training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform standardization on split data. Standardization is performed on independent features
    # only i.e. X_train/X_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Available models
    models = {
        "Support Vector Machines": svm.SVC(**svm_params),
        "Logistic Regression": LogisticRegression(**lr_params),
        "K-Nearest Neighbor": KNeighborsClassifier(**knn_params),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(algorithm='SAMME'),
        "Gradient Boost": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(),
        "Gaussian NB": GaussianNB(),

    }

    # Get the machine learning model selected by user
    model = models[model_name]

    # Train the model
    model.fit(X_train, y_train)

    # Prediction for test data
    y_pred = model.predict(X_test)

    # Predict probability
    y_pred_proba = model.predict_proba(X_test)

    # Create data frame for storing model performance parameters
    performance_tests = ['Accuracy', 'F1 Score', 'Precision', 'Recall',
                         'ROC AUC Score']  # This will be the name of columns
    df_performance_metric = pd.DataFrame(columns=performance_tests)

    if label == 'Multi Class':
        # Get the performance metrics and add them in the data frame
        df_performance_metric.loc[0, 'Accuracy'] = "{:.2f}".format(accuracy_score(y_test, y_pred))
        df_performance_metric.loc[0, 'F1 Score'] = "{:.2f}".format(f1_score(y_test, y_pred, average='weighted'))
        df_performance_metric.loc[0, 'Precision'] = "{:.2f}".format(precision_score(y_test, y_pred, average='weighted'))
        df_performance_metric.loc[0, 'Recall'] = "{:.2f}".format(recall_score(y_test, y_pred, average='weighted'))
        df_performance_metric.loc[0, 'ROC AUC Score'] = "{:.2f}".format(roc_auc_score(y_test, y_pred_proba,
                                                                                      average='weighted',
                                                                                      multi_class='ovr'))

    else:
        # Get the performance metrics and add them in the data frame
        df_performance_metric.loc[0, 'Accuracy'] = "{:.2f}".format(accuracy_score(y_test, y_pred))
        df_performance_metric.loc[0, 'F1 Score'] = "{:.2f}".format(f1_score(y_test, y_pred))
        df_performance_metric.loc[0, 'Precision'] = "{:.2f}".format(precision_score(y_test, y_pred))
        df_performance_metric.loc[0, 'Recall'] = "{:.2f}".format(recall_score(y_test, y_pred))
        df_performance_metric.loc[0, 'ROC AUC Score'] = "{:.2f}".format(roc_auc_score(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, df_performance_metric, cm


# Function to predict outcome given the input data and based on trained machine learning model
def model_predictions(input_data, model, scaler):
    # Convert the input data into a 2D array. This is required for machine learning model
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Scale the input data. This is required before making prediction
    input_array_scaled = scaler.transform(input_array)

    # Get prediction from the trained model
    prediction = model.predict(input_array_scaled)

    # Get prediction probability
    probability = model.predict_proba(input_array_scaled)

    return prediction, probability
