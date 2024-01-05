import shutil
import os
import json
import pickle
import numpy as np 
import pandas as pd 
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
import copy
import pickle
# Classifier Libraries
from sklearn.ensemble import IsolationForest
import collections
from sklearn.svm import  SVC
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
# train/test split libraries and preprocessing
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, balanced_accuracy_score, cohen_kappa_score, average_precision_score
from collections import Counter
import pandas as pd
from flask import Flask, render_template, request, send_file
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
df2 = pd.read_csv("synthetic.csv")[:100000]
# df_train = pd.read_csv("synthetic.csv")[-100000:]
time_till_train_start = 1000
TRAIN_EVERY = 1000

if 'forest_train_over' in os.listdir():
    os.rmdir('forest_train_over')
    
clf_loaded = None
with open('model_xgbc.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)


init_dataset = True

def start_xgboost_train():
    os.mkdir('forest_train_start')
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    # print(uploaded_file, type(uploaded_file))
    if uploaded_file.filename != '':
        if init_dataset:
            uploaded_file.save("synthetic.csv")
            init_dataset = False
            start_xgboost_train()
            return "1st csv, Ping me for inference ready until ready"
            
        else:
            uploaded_file.save("temp.csv")
            df1 = pd.read_csv('temp.csv')
            df2 = pd.merge(df1, df2, on='merge_key')
            
            with open('synthetic.csv', 'w', encoding='utf-8') as f:
                df2.to_csv(f, index=False)
                
            start_xgboost_train()
            return "Joined csv, Ping me for inference ready until ready"
     
     
     
@app.route('/upload_list', methods=['POST', 'GET'])
def upload_list():
    global df2, clf_loaded
    content = request.json
    
    # df2 = pd.read_csv('synthetic.csv')
    content['data']['step'] = df2['step'].values[-1]+1
    
    for i in list(content['data'].keys()):
        content['data'][i] = [content['data'][i]]
    
    df1 = pd.DataFrame.from_dict(content['data'])
    # merged_df = pd.merge(df1, df2, on='merge_key')

    df2 = pd.concat([df2, df1], ignore_index=True)
    
    with open('synthetic.csv', 'w', encoding='utf-8') as f:
        df2.to_csv(f, index=False)
    
    try:
        start_xgboost_train()
    except:
        print("Loaded for future")
    




    # pprint(df1)
    df = df1.copy()
    df = df.rename(columns = {'nameOrig' : 'origin', 'oldbalanceOrg' : 'sender_old_balance', 'newbalanceOrig': 'sender_new_balance', 'nameDest' : 'destination', 'oldbalanceDest' : 'receiver_old_balance', 'newbalanceDest': 'receiver_new_balance', 'isFraud' : 'isfraud'})
    df = df.drop(columns = ['step', 'isFlaggedFraud'], axis = 'columns')
    cols = df.columns.tolist()
    new_position = 3

    cols.insert(new_position, cols.pop(cols.index('destination')))
    df = df[cols]
    data = df.copy()
    data['type2'] = np.nan
    data.loc[df.origin.str.contains('C') & df.destination.str.contains('C'), 'type2'] = 'CC'
    data.loc[df.origin.str.contains('C') & df.destination.str.contains('M'), 'type2'] = 'CM'
    data.loc[df.origin.str.contains('M') & df.destination.str.contains('C'), 'type2'] = 'MC'
    data.loc[df.origin.str.contains('M') & df.destination.str.contains('C'), 'type2'] = 'MM'
    cols = data.columns.tolist()
    new_position = 1
    
    # pprint(data)

    cols.insert(new_position, cols.pop(cols.index('type2')))
    data = data[cols]
    data.drop(columns = ['origin','destination'], axis = 'columns', inplace = True)

    # pprint(data)

    # data_ = {'type': ['type_CASH_OUT', 'type_DEBIT','type_PAYMENT', 'type_TRANSFER', 'type2_CM']}
    # df = pd.DataFrame(data_)

    # Define payment types
    payment_types = ['type_CASH_OUT', 'type_DEBIT','type_PAYMENT', 'type_TRANSFER', 'type2_CM']

    # Apply a function to create boolean columns
    for payment_type in payment_types:
        data[payment_type] = data['type'].values[0] == payment_type.split('_')[-1]

    data = data.drop(columns=['type','type2'], axis=1)
    print(data.keys())
    # if 'class' in list(data.keys()):
    #     data.drop(columns=['class'], axis=1)
        
    # print(list(data.keys()), 'class' in list(data.keys()))
    X = data.drop(columns=['isfraud'])

    sc = StandardScaler()
    X = sc.fit_transform(X)

    predictions = clf_loaded.predict(X)
    # print(content['data']['step'][0], type(content['data']['step'][0]))
    
    return {"idx":content['data']['step'][0].tolist(), "pred":int(predictions.tolist()[0]) }       
        

@app.route('/train_status', methods=['GET'])
def train_status():
    global df2, clf_loaded
    # if init_dataset:
    #     return "Upload csv first"
    
    if 'forest_train_over' in os.listdir():
        os.rmdir('forest_train_over')
        # load new model <=====
        with open('model_xgbc.pkl', 'rb') as f:
            clf_loaded = pickle.load(f)

        return "Train Over"
    
    else:
        return "Not Over"
    

from pprint import pprint

@app.route('/inference', methods=['GET'])
def inference():
    global df2, clf_loaded
    if 'forest_train_over' in os.listdir():
        os.rmdir('forest_train_over')
        with open('model_xgbc.pkl', 'rb') as f:
            clf_loaded = pickle.load(f)
            
            

    # pprint(df1)
    df = df2.copy()
    df = df.rename(columns = {'nameOrig' : 'origin', 'oldbalanceOrg' : 'sender_old_balance', 'newbalanceOrig': 'sender_new_balance', 'nameDest' : 'destination', 'oldbalanceDest' : 'receiver_old_balance', 'newbalanceDest': 'receiver_new_balance', 'isFraud' : 'isfraud'})
    df = df.drop(columns = ['step', 'isFlaggedFraud'], axis = 'columns')
    cols = df.columns.tolist()
    new_position = 3

    cols.insert(new_position, cols.pop(cols.index('destination')))
    df = df[cols]
    data = df.copy()
    data['type2'] = np.nan
    data.loc[df.origin.str.contains('C') & df.destination.str.contains('C'), 'type2'] = 'CC'
    data.loc[df.origin.str.contains('C') & df.destination.str.contains('M'), 'type2'] = 'CM'
    data.loc[df.origin.str.contains('M') & df.destination.str.contains('C'), 'type2'] = 'MC'
    data.loc[df.origin.str.contains('M') & df.destination.str.contains('C'), 'type2'] = 'MM'
    cols = data.columns.tolist()
    new_position = 1
    
    # pprint(data)

    cols.insert(new_position, cols.pop(cols.index('type2')))
    data = data[cols]
    data.drop(columns = ['origin','destination'], axis = 'columns', inplace = True)

    # pprint(data)

    # data_ = {'type': ['type_CASH_OUT', 'type_DEBIT','type_PAYMENT', 'type_TRANSFER', 'type2_CM']}
    # df = pd.DataFrame(data_)

    # Define payment types
    payment_types = ['type_CASH_OUT', 'type_DEBIT','type_PAYMENT', 'type_TRANSFER', 'type2_CM']

    # Apply a function to create boolean columns
    for payment_type in payment_types:
        data[payment_type] = data['type'].values[0] == payment_type.split('_')[-1]

    data = data.drop(columns=['type','type2'], axis=1)
    print(data.keys())
    # if 'class' in list(data.keys()):
    #     data.drop(columns=['class'], axis=1)
        
    # print(list(data.keys()), 'class' in list(data.keys()))
    X = data.drop(columns=['isfraud'])

    sc = StandardScaler()
    X = sc.fit_transform(X)


    predictions = clf_loaded.predict(X)
    df1 = pd.DataFrame(list(zip([i for i in range(len(predictions))], predictions)),
              columns=['step','isFraud'])

    # inference new model <=====
    # df1 = pd.read_csv("output.csv")
    # df1['step'].values = [i+1 for i in range(len(df1['step'].values))]
    
    # df2['step'].values = [i+1 for i in range(len(df2['step'].values))]

    all_poss = []
    for i in df2:
        if np.issubdtype(df2[i].dtype, np.number) and i!="isFraud":
            all_poss.append(i)
    

    if len(all_poss) == 0:
        all_poss.append(list(df2)[0])
        
    col = all_poss[len(all_poss)//2]
    extracted_col = df2[col]
    df1 = pd.concat([df1, extracted_col], axis=1)
    # pprint(df1)
    # with open('output.csv', 'w', encoding='utf-8') as f:
    #     df1.to_csv(f, index=False)    
    
    x = json.loads(df1.to_json(orient="split"))
    y = {}
    # pprint(x)
    # print(x.keys(), x['columns'])
    # index = [8, 9]
    for i in x['columns']:
        y[i] = []
    
    cols = list(y.keys())
 
    # pprint(cols)

    for i in x['data']:
        for j in range(0,3):
            y[cols[j]].append(i[j])
            
    # pprint(y)
    return y


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)