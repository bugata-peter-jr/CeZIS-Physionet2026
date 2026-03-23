#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from tqdm import tqdm

from helper_code import *

import pandas as pd

################################################################################
# Path & Constant Configuration (Added for Robustness)
################################################################################

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file relative to the script location
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')
        

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    mddf = pd.read_csv(patient_data_file)
    
    features = ['Age_cat','Sex_cat','BMI_cat']
    
    # transforming sex
    mddf['Sex_cat'] = mddf.Sex.map({'Male':1, 'Female':0})
    
    # discertizing  age
    mddf = discretize_age(mddf)   

    mddf.set_index('BDSPPatientID', inplace=True)
    
    mddf_pos = mddf.query("Cognitive_Impairment == True")
    
    mddf_neg = mddf.query("Cognitive_Impairment == False")
    
    mddf_pos_train = mddf_pos.query("BMI == BMI")
    mddf_pos_test = mddf_pos.query("BMI != BMI")
    
    y1_train = mddf_pos_train.pop("BMI").values
    
    mddf_pos_train = mddf_pos_train.loc[:,['Age_cat','Sex_cat']]
    mddf_pos_test = mddf_pos_test.loc[:,['Age_cat','Sex_cat']]
    
    X1_train = mddf_pos_train.values
    X1_test = mddf_pos_test.values
    
    rf_reg_p = RandomForestRegressor(n_estimators=1000)
    rf_reg_p.fit(X1_train, y1_train)
    pos_pred = rf_reg_p.predict(X1_test)
    
    mddf_neg_train = mddf_neg.query("BMI == BMI")
    mddf_neg_test = mddf_neg.query("BMI != BMI")
    
    y2_train = mddf_neg_train.pop("BMI").values
    
    mddf_neg_train = mddf_neg_train.loc[:,['Age_cat','Sex_cat']]
    mddf_neg_test = mddf_neg_test.loc[:,['Age_cat','Sex_cat']]

    X2_train = mddf_neg_train.values
    X2_test = mddf_neg_test.values    
    
    rf_reg_n = RandomForestRegressor(n_estimators=1000)
    rf_reg_n.fit(X2_train, y2_train)
    neg_pred = rf_reg_n.predict(X2_test)
    
    mddf.loc[mddf_pos_test.index,'BMI'] = pos_pred
    mddf.loc[mddf_neg_test.index,'BMI'] = neg_pred
    
    # discretizing BMI
    mddf = discretize_bmi(mddf)
    
    y_train = mddf.pop('Cognitive_Impairment').values
    
    # only used features
    X_train = mddf.loc[:,features].values
    
    clf = RandomForestClassifier(random_state=2026, n_estimators=1000, class_weight={False:8, True:1})
    clf.fit(X_train, y_train)
    
    # second classifier without BMI
    X_train_r = mddf.loc[:,['Age_cat','Sex_cat']].values
    clf_without_bmi = RandomForestClassifier(random_state=2026, n_estimators=1000, class_weight={False:8, True:1})
    clf_without_bmi.fit(X_train_r, y_train)    

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    model = {'clf':clf, 'clf_without_bmi': clf_without_bmi}

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(model, record, data_folder, verbose):
    
    # load the model
    clf = model.get('clf')
    clf_without_bmi = model.get('clf_without_bmi')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    mddf = pd.read_csv(patient_data_file)

    # Extract identifiers from the record dictionary
    patient_id_col = HEADERS['bids_folder']
    session_id_col = HEADERS['session_id']

    patient_id = record[patient_id_col]
    session_id = record[session_id_col]

    mask = (mddf[patient_id_col] == patient_id) & (mddf[session_id_col] == session_id)    
    mddf = mddf.loc[mask]
    
    # imputing sex (default Male)
    idx = mddf.query("Sex not in ['Male','Female']").index
    mddf.loc[idx, 'Sex'] = 'Male'

    # transforming sex
    mddf['Sex_cat'] = mddf.Sex.map({'Male':1, 'Female':0})

    # discertizing  age
    mddf = discretize_age(mddf) 

    # according to BMI, we use first or second classifier    
    bmi = mddf.BMI.iat[0]    
    if np.isnan(bmi):
        features = ['Age_cat','Sex_cat']
        clf_tmp = clf_without_bmi
    else:
        # discretizing BMI
        mddf = discretize_bmi(mddf)
        features = ['Age_cat','Sex_cat','BMI_cat']
        clf_tmp = clf
        
    X_test = mddf.loc[:,features].values
        
    binary_output = clf_tmp.predict(X_test)[0]
    probability_output = clf_tmp.predict_proba(X_test)[0][1]

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################



# Save your trained model.
def save_model(model_folder, model):
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(model, filename, protocol=0)
 
# discertize age
def discretize_age(mddf):
    mddf['Age_cat'] = -1
    mddf.loc[mddf.query("Age >= 50 and Age < 60").index,'Age_cat'] = 0
    mddf.loc[mddf.query("Age >= 60 and Age < 70").index,'Age_cat'] = 1
    mddf.loc[mddf.query("Age >= 70 and Age < 80").index,'Age_cat'] = 2
    mddf.loc[mddf.query("Age >= 80").index,'Age_cat'] = 3
    return mddf

# disceretize bmi
def discretize_bmi(mddf):
    mddf['BMI_cat'] = -1 # NULL (should not be present)
    mddf.loc[mddf.query("BMI < 25").index,'BMI_cat'] = 0 # underweight and healthy weight
    mddf.loc[mddf.query("BMI >= 25 and BMI < 30").index,'BMI_cat'] = 1 # overweight
    mddf.loc[mddf.query("BMI >= 30 and BMI < 35").index,'BMI_cat'] = 2 # class 1 obesity
    mddf.loc[mddf.query("BMI >= 35 and BMI < 40").index,'BMI_cat'] = 3 # class 2 obesity
    mddf.loc[mddf.query("BMI >= 40").index,'BMI'] = 4 # severe/class 3 obesity
    return mddf