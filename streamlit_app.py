import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

st.title("Hello ML Learners")
st.markdown(
    """
    This is ML Learning.    
    """
)

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

global X_test

if uploaded_file is not None:
    X_test = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded.")
    
    option = st.selectbox(
        "Choose an option:",
        ["Logistic Regression", "Decision Tree Classifier", "K-Nearest Neighbour Classifier", "Naive Bayes Classifier - Gaussian", "Ensemble Model - Random Forest", "Ensemble Model - XGBoost"]
    )

    st.write("You selected:", option)  

    y_test = pd.read_csv("y_test/y_test.csv")

    if option == "Logistic Regression":
        st.write("Running Logistic Regression")
        logistic_model = pickle.load(open("model/logistic_model.pkl", "rb"))
    
        y_pred = logistic_model.predict(X_test)
    
        acc = accuracy_score(y_test, y_pred)
        acc = acc* 100
    
        y_prob = logistic_model.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(y_test, y_prob)
    
        mcc = matthews_corrcoef(y_test, y_pred)
    
        conf_matrix = confusion_matrix(y_test, y_pred) 
        TN, FP, FN, TP = conf_matrix.ravel()
    
        precision = TP / ( TP + FP )
   
        recall = TP / ( TP + FN )
    
        f1score = ( 2 * ( precision * recall ) ) / ( precision + recall )
   
        # Sample DataFrame
        data = pd.DataFrame({
        'ML Model Name': ['Logistic Regression'],
        'Accuracy': [acc],
        'AUC': [auc_score],
        'Precision':[precision],
        'Recall':[recall],
        'F1':[f1score],
        'MCC':[mcc]
        })
        
        st.write("Evaluation Metrics : ")
        st.table(data)
    
        st.write("Confusion Matrix : ")
        st.write("True Negative : ", TN)
        st.write("False Positive : ", FP)
        st.write("False Negative : ", FN)
        st.write("True Positive : ", TP)
    
    
    elif option == "Decision Tree Classifier":
        st.write("Running Decision Tree Classifier")
        dt_model = pickle.load(open("model/dt_model.pkl", "rb"))
    
        y_pred = dt_model.predict(X_test)
    
        acc = accuracy_score(y_test, y_pred)
        acc = acc* 100
    
        y_prob = dt_model.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(y_test, y_prob)
    
        mcc = matthews_corrcoef(y_test, y_pred)
    
        conf_matrix = confusion_matrix(y_test, y_pred) 
        TN, FP, FN, TP = conf_matrix.ravel()
    
        precision = TP / ( TP + FP )
   
        recall = TP / ( TP + FN )
    
        f1score = ( 2 * ( precision * recall ) ) / ( precision + recall )
   
        # Sample DataFrame
        data = pd.DataFrame({
            'ML Model Name': ['Decision Tree Classifier'],
            'Accuracy': [acc],
            'AUC': [auc_score],
            'Precision':[precision],
            'Recall':[recall],
            'F1':[f1score],
            'MCC':[mcc]
        })
        st.write("Evaluation Metrics : ")
        st.table(data)
    
        st.write("Confusion Matrix : ")
        st.write("True Negative : ", TN)
        st.write("False Positive : ", FP)
        st.write("False Negative : ", FN)
        st.write("True Positive : ", TP)
    
    elif option == "K-Nearest Neighbour Classifier":
        st.write("K-Nearest Neighbour Classifier")
        knn_model = pickle.load(open("model/knn_model.pkl", "rb"))
        
        y_pred = knn_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        acc = acc* 100
        
        y_prob = knn_model.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(y_test, y_prob)
        
        mcc = matthews_corrcoef(y_test, y_pred)
        
        conf_matrix = confusion_matrix(y_test, y_pred) 
        TN, FP, FN, TP = conf_matrix.ravel()
        
        precision = TP / ( TP + FP )
   
        recall = TP / ( TP + FN )
    
        f1score = ( 2 * ( precision * recall ) ) / ( precision + recall )
   
        # Sample DataFrame
        data = pd.DataFrame({
        'ML Model Name': ['K-Nearest Neighbour Classifier'],
        'Accuracy': [acc],
        'AUC': [auc_score],
        'Precision':[precision],
        'Recall':[recall],
        'F1':[f1score],
        'MCC':[mcc]
        })
        st.write("Evaluation Metrics : ")
        st.table(data)   
        st.write("Confusion Matrix : ")
        st.write("True Negative : ", TN)
        st.write("False Positive : ", FP)
        st.write("False Negative : ", FN)
        st.write("True Positive : ", TP)
        
        
    elif option == "Naive Bayes Classifier - Gaussian":
        st.write("Running Naive Bayes Classifier - Gaussian")
        nb_model = pickle.load(open("model/nb_model.pkl", "rb"))
    
        y_pred = nb_model.predict(X_test)
    
        acc = accuracy_score(y_test, y_pred)
        acc = acc* 100
    
        y_prob = nb_model.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(y_test, y_prob)
    
        mcc = matthews_corrcoef(y_test, y_pred)
    
        conf_matrix = confusion_matrix(y_test, y_pred) 
        TN, FP, FN, TP = conf_matrix.ravel()
    
        precision = TP / ( TP + FP )
   
        recall = TP / ( TP + FN )
    
        f1score = ( 2 * ( precision * recall ) ) / ( precision + recall )
   
        # Sample DataFrame
        data = pd.DataFrame({
        'ML Model Name': ['Naive Bayes Classifier - Gaussian'],
        'Accuracy': [acc],
        'AUC': [auc_score],
        'Precision':[precision],
        'Recall':[recall],
        'F1':[f1score],
        'MCC':[mcc]
        })
        st.write("Evaluation Metrics : ")
        st.table(data)   
        st.write("Confusion Matrix : ")
        st.write("True Negative : ", TN)
        st.write("False Positive : ", FP)
        st.write("False Negative : ", FN)
        st.write("True Positive : ", TP)
        
    elif option == "Ensemble Model - Random Forest":
        st.write("Running Random Forest Classifier")
        rf_classifier_model = pickle.load(open("model/rf_classifier_model.pkl", "rb"))
        
        y_pred = rf_classifier_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        acc = acc* 100
    
        y_prob = rf_classifier_model.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(y_test, y_prob)
    
        mcc = matthews_corrcoef(y_test, y_pred)
    
        conf_matrix = confusion_matrix(y_test, y_pred) 
        TN, FP, FN, TP = conf_matrix.ravel()
    
        precision = TP / ( TP + FP )
   
        recall = TP / ( TP + FN )
    
        f1score = ( 2 * ( precision * recall ) ) / ( precision + recall )
   
        # Sample DataFrame
        data = pd.DataFrame({
        'ML Model Name': ['Ensemble Model - Random Forest'],
        'Accuracy': [acc],
        'AUC': [auc_score],
        'Precision':[precision],
        'Recall':[recall],
        'F1':[f1score],
        'MCC':[mcc]
        })
        
        st.write("Evaluation Metrics : ")
        st.table(data)   
        st.write("Confusion Matrix : ")
        st.write("True Negative : ", TN)
        st.write("False Positive : ", FP)
        st.write("False Negative : ", FN)
        st.write("True Positive : ", TP)
        
    elif option == "Ensemble Model - XGBoost":
        st.write("Running Ensemble Model - XGBoost")
        xg_boost_model = pickle.load(open("model/xg_boost_model.pkl", "rb"))
        
        y_pred = xg_boost_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        acc = acc* 100
        
        y_prob = xg_boost_model.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(y_test, y_prob)
        
        mcc = matthews_corrcoef(y_test, y_pred)
        
        conf_matrix = confusion_matrix(y_test, y_pred) 
        TN, FP, FN, TP = conf_matrix.ravel()
    
        precision = TP / ( TP + FP )
   
        recall = TP / ( TP + FN )
    
        f1score = ( 2 * ( precision * recall ) ) / ( precision + recall )
        
        # Sample DataFrame
        data = pd.DataFrame({
        'ML Model Name': ['Ensemble Model - XGBoost'],
        'Accuracy': [acc],
        'AUC': [auc_score],
        'Precision':[precision],
        'Recall':[recall],
        'F1':[f1score],
        'MCC':[mcc]
        })
        
        st.write("Evaluation Metrics : ")
        st.table(data)   
        st.write("Confusion Matrix : ")
        st.write("True Negative : ", TN)
        st.write("False Positive : ", FP)
        st.write("False Negative : ", FN)
        st.write("True Positive : ", TP)





