import pickle 
import os
import pandas as pd
from joblib import dump, load
import time

while True:
    if 'forest_train_start' in os.listdir():
        print("TRAIN STARTED")
        os.rmdir('forest_train_start')
        model = None
        #### model training
        time.sleep(10)
        dump(model, 'forest_train_start.joblib') 
        os.mkdir('forest_train_over')
        print("TRAIN OVER")
        
        