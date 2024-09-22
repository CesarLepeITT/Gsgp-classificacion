from cgi import print_environ
from ntpath import join
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, make_union
import sys
import os
import subprocess
import pandas as pd
import numpy as np
import time
import math
from sympy import *
import shutil
from re import findall
import re
import datetime

this_dir = os.path.dirname(os.path.realpath(__file__))
global vars


class gsgpcudaregressor(BaseEstimator):
    global fin
    global fin_dos
    global nvar
    # method to initialize the class
    def __init__(self,  g=1024, pop_size=1024, max_len=10, func_ratio=0.5,
                 variable_ratio=0.5, max_rand_constant=10, sigmoid=0,
                 error_function=0, oms=0, normalize=0, do_min_max=0, protected_division=0):
        env = dict(os.environ)

        self.g = g
        self.pop_size = pop_size
        self.max_len = max_len
        self.func_ratio = func_ratio
        self.variable_ratio = variable_ratio
        self.max_rand_constant = max_rand_constant
        self.sigmoid = sigmoid
        self.error_function = error_function
        self.oms = oms
        self.normalize = normalize
        self.do_min_max = do_min_max
        self.protected_division = protected_division
        self.exe_name = 'GsgpCuda.x'
        self.name_run1 = str(np.random.randint(2**15-1))
        self.name_ini = ''
        self.log_path = this_dir + '/' + self.name_run1 + '/'
        self.fin=''
        self.fin_dos=''
        self.nvar =0

        text = '''numberGenerations={}
populationSize={}
maxIndividualLength={}
functionRatio={}
variableRatio={}
maxRandomConstant={}
sigmoid={}
errorFunction={}
oms={}
normalize={}
do_min_max={}
protected_division={}
logPath={}
'''.format(self.g, self.pop_size, self.max_len, self.func_ratio, self.variable_ratio,
          self.max_rand_constant, self.sigmoid, self.error_function,
          self.oms, self.normalize, self.do_min_max, self.protected_division ,self.log_path)

        # Create dataPath for Data Files
        os.makedirs(self.log_path, exist_ok=True)


        test = os.listdir(this_dir)
        for item in test:
            if item.endswith(".csv"):
                os.remove(os.path.join(this_dir, item))


        self.name_ini = self.log_path + self.name_run1 + "_configuration.ini"
        ffile = open(self.name_ini, "w")
        ffile.write(text)
        time.sleep(1)

    def fit(self, X_train, y_train, sample_weight=None):
        self.X_train = X_train
        self.y_train = y_train

        data = pd.DataFrame(self.X_train)
        data['target'] = self.y_train
        data.to_csv(this_dir + '/' + "train.csv",
                    header=None, index=None, sep='\t')
        time.sleep(1)
        self.nvar = data.shape[1] -1

        subprocess.call(' '.join([this_dir + '/' + self.exe_name,
                        '-train_file ' + this_dir + '/' + 'train.csv',
                        " -output_model " + self.name_run1,
                        '-log_path ' + self.name_ini]),
                        shell=True, cwd=this_dir)

        time.sleep(1)


    def predict(self, X_test):
        hora_actual = datetime.datetime.now()
        hora_actual_str = hora_actual.strftime('%H:%M:%S')
        print('La hora desde predit es:', hora_actual_str)
        self.X_test = X_test
        data = pd.DataFrame(self.X_test)
        data.to_csv(this_dir + '/' + "unseen_data.csv",
                    header=None, index=None, sep='\t')

        time.sleep(1)

        subprocess.call(' '.join([this_dir + '/' + self.exe_name,
                                  '-model ' + self.name_run1,
                                  '-input_data ' + this_dir + '/' + 'unseen_data.csv',
                                  '-prediction_output ' + self.name_run1 + '_prediction.csv',
                                  '-log_path ' + self.name_ini]),
                        shell=True, cwd=this_dir)

        time.sleep(1)

        # read the prediction file
        name_prediction = self.log_path + self.name_run1 + "_prediction.csv"
        y_pred = []
        with open(name_prediction, 'r') as f:
            for line in f:
                y_pred.append(float(line.strip()))

        return y_pred

    def check_valid_expression(self, expr_str):
        try:
            expr = sympify(expr_str)
            return True
        except SympifyError:
            return False

    def best_individual(self, model):
        tmp = self.log_path + self.name_run1 + "_ModelExpression.csv"
        with open(tmp) as f:
            contents = f.read()

        con_split = contents.split(",")[:-1]
        model_final = ""
        model_f = ""
        p = 0
        if(model == 0):
            model_final = self.fin_dos
        else:
            model_final = self.fin

        return model_final

    def get_model(self):
        return self.best_individual(self)

    def get_n_nodes(self, model):
        class protected_division(Function):
            @classmethod
            def eval(cls, x, y):
                if y.is_Number:
                    if y.is_zero:
                        return x/sqrt(1+(y**2))
                    else:
                        return x/y

        def convert_protected_division(expr):
          pattern = re.compile(r'protected_division\((.+?),\s*(.+?)\)')
          while True:
            match = pattern.search(expr)
            if match is None:
                break
            numer, denom = match.groups()
            if "protected_division" in numer or "protected_division" in denom:
                expr = expr[:match.start()] + \
                    f"({numer})/({denom})" + expr[match.end():]
            else:
                numer = numer.replace("'", "")
                denom = denom.replace("'", "")
                expr = expr[:match.start()] + \
                f"({numer})/({denom})" + expr[match.end():]
          return expr
        
        tmp = self.log_path + self.name_run1 + "_ModelExpression.csv"
        with open(tmp) as f:
            contents = f.read()

        j = 0
        t = 0
        con_split = contents.split(":")[:-1]
        model_final = ''
        a = ''
        m = ''
        p = 0
        b = ''
        var_names=[]
        for i in range(self.nvar):
          var = f"X_{i}"
          var_names.append(var)

        symbol = [s for s in var_names if s.startswith('X_') or s.startswith('X')]
        protected = {'protected_division': True}
        if model == 0:
            for sub_exp in (con_split):
              j = 0
              expr = sympify(sub_exp, {'protected_division': protected_division, **{s: symbols(s) for s in symbol}},evaluate=False)     
              test_p = convert_protected_division(str(expr))
              n = sympify(test_p,evaluate=False)
              for arg in preorder_traversal(n):
                j += 1
              
              t += j
              if(p == 0):
                m = str(test_p)
              else:
                m += '+' + str(test_p)
            
              p += 1

            if(m==''):
              self.fin_dos = m 
            if(self.fin_dos is None or ''):
              self.fin_dos = m 
            else:
              self.fin_dos = m

        else:
            for sub_exp in (con_split):
              j = 0
              expr = sympify(sub_exp, {'protected_division': protected_division, **{s: symbols(s) for s in symbol}},evaluate=True)
              test_p = convert_protected_division(str(expr))
              testy = sympify(test_p,evaluate=True)
              if(p == 0):
                m = str(testy)
              else:
                m += '+' + str(testy)
            
              p += 1

            
            l = sympify(m,evaluate=True)
            
            for arg in preorder_traversal(l):
              j += 1

            t = j
            self.fin = l
        return t
