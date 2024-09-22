import gsgpcuda
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing 
import random
import numpy as np
import math
import os
import shutil

# Directorio en el que deseas buscar la carpeta más reciente
directorio = '/home/tree/Desktop/NORMALIZE/'


#path of problems
path =  ["/home/tree/Desktop/NORMALIZE/Concrete.txt",
        "/home/tree/Desktop/NORMALIZE/EColing.txt",
        "/home/tree/Desktop/NORMALIZE/EHeating.txt",
        "/home/tree/Desktop/NORMALIZE/housing.txt",
        "/home/tree/Desktop/NORMALIZE/yacht_hydrodynamics.txt",
        "/home/tree/Desktop/NORMALIZE/tower.txt"]

new_folder_path = ""
tmp = ""
tmp_2 =""
tmp_folder=""
for file_path in path:
    df = pd.DataFrame()
    filename = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path, header=None, sep='\s+')
    df_predictions_of_run = pd.DataFrame()
    df_fitness_train = pd.DataFrame()
    data_test =pd.DataFrame()
    data_trace =pd.DataFrame()
    
    print(filename)
    filename_tmp = directorio + filename
    if not os.path.exists(filename_tmp):
        os.mkdir(filename_tmp)

    # Define row and columns of dataset
    nrow = len(df.index)
    nvar = df.shape[1]

    #Separate data of target
    X = df.iloc[0:nrow, 0:nvar-1]

    #load colum of target
    y = df.iloc[:nrow, nvar-1]

    repeat = 30
    rmse_abs_test = [None]*repeat
    new_error_mean_precentage = [None]*repeat
    rmse_test =[None]*repeat
    mse_test =[None]*repeat
    r2_test=[None]*repeat
    fitness_train  = [None]*repeat
    modelo_previo = [None]*repeat
    model_size_previo = [None]*repeat
    models  = [None]*repeat
    model_size  = [None]*repeat
    medi_static_train = [None]*4
    medi_static_test =  [None]*4
    medi_static_trace_pos = [None]*4
    medi_static_trace_neg =  [None]*4
    stats_df_train  = [None]*4
    stats_df_predict  = [None]*4
    # Inicializa una lista para almacenar los recuentos de "-1"
    recuentos_minus_uno = []
    # Inicializa una lista para almacenar los recuentos de "1"
    recuentos_pos_uno = []

    #this cyclo repeat repeat.
    for i in range(repeat):
        print("Number of Run \n",i)
        est = gsgpcuda.gsgpcudaregressor(
            g=200,
            pop_size=1024,
            max_len=1024,
            func_ratio=0.5,
            variable_ratio=0.5,
            max_rand_constant=10,
            sigmoid = 1,
            error_function=0,
            oms=1,
            normalize=1,
            do_min_max=2,
            protected_division=0
        )
        # Split data 
        n = random.randint(0,9000)
        X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.70,test_size=0.30,random_state=n)
        
        #train model
        est.fit(X_train,y_train)
        #est.fit(X_scaled,Y_scaled)
        
        #prediction with de model
        values_predict = est.predict(X_test)
        df_predictions_of_run[i] = values_predict

        #obtein size or complexity of model generate with gsgp
        model_size_previo[i] = est.get_n_nodes(0)
        print("Segunda vuelta para contar mdelo RED -*/-*/-*/-*/-*/-*/-*/-*/-*/-*/-*/\n")
        model_size[i] = est.get_n_nodes(1)
        print("paso la complejidad \n")          
        
        #obtein the model generate with gsgp
        modelo_previo[i] = est.best_individual(0)
        models[i] = est.best_individual(1)
        print("paso el modelo \n")          
        
        