"""
@author: Jacopo Braccio s273999
"""


import pandas as pd
import numpy as np
import functions as fc


"""Preparation of the dataset. During this phase a dataset is created by importing the datas from the csv file.
The dataset will be splitted into three subsets(training subset - 50%, validation subset - 25%, test subset - 50%)"""

x = pd.read_csv ("parkinsons_updrs.csv") #Creation of the DataFrame x by importing the csv file.
np.random.seed(11) #A random seed allows to generate always the same shuffled matrix when we launch the program
x_shuffled = x.sample(frac=1).reset_index(drop=True) #shuffle the dataset according to the seed

#CREATING SUBSETS 
x_tr = x_shuffled[0:2938] #training dataset
x_val = x_shuffled[2938:4407] #validation dataset 
x_test = x_shuffled[4407:] #test dataset

#total UPDRS for training set
y_tr = x_tr['total_UPDRS']
y_tr_mean = y_tr.mean() #mean value of totalUPDRS for training set
y_tr_var = y_tr.std() #std value of totalUPDRS for training set
y_tr_norm = (y_tr - y_tr_mean)/(y_tr_var) #normalization of the solution of the training
y_test = x_test ['total_UPDRS'] #total UPDRS for test set
y_val = x_val ['total_UPDRS'] #total UPDRS for validation set
y_val = y_val.values
y_test_values = y_test.values
y_val_norm = (y_val - y_tr_mean)/(y_tr_var)
y_test_norm = (y_test - y_tr_mean)/y_tr_var
y_test_norm = y_test_norm.values

#dropping unuseful features
x_tr = x_tr.drop(['subject#','test_time','total_UPDRS'], axis = 1)
x_test = x_test.drop(['subject#','test_time','total_UPDRS'], axis = 1) 
x_val = x_val.drop(['subject#','test_time','total_UPDRS'], axis = 1)          

features = list(x_tr.columns)

#normalization of the dataset with mean and std of the training set                   
x_tr_norm = (x_tr - x_tr.mean())/(x_tr.std())
x_test_norm = (x_test - x_tr.mean())/(x_tr.std())
x_val_norm = (x_val - x_tr.mean())/(x_tr.std())

#convert into nparrays
x_tr_norm = x_tr_norm.values
x_test_norm = x_test_norm.values
x_val_norm = x_val_norm.values
y_tr_norm = y_tr_norm.values
""""""

"""LLS Algorithm"""
reg_LLS = fc.SolveLLS(y_tr_norm, x_tr_norm, y_test_values, x_test_norm, y_tr_mean , y_tr_var)
reg_LLS.run()
reg_LLS.print_result("'''Linear Least Square'''")
reg_LLS.plot_w(features,"LLS optimal soution vector: ")
reg_LLS.plot_estimation("LLS estimated solution vs real solution:")
reg_LLS.plot_Histograms("LLS: histograms of the estimation error: ")
reg_LLS.mean_std_train("LLS mean and standard deviation of regression errors for training dataset: ")
reg_LLS.mean_std_test("LLS mean and standard deviation of regression errors for test dataset: ")
reg_LLS.MSETraining("LLS MSE of the training set: ")
reg_LLS.MSETest("LLS MSE of the test set: ")
reg_LLS.r_squared("Coefficiente di determinazione R^2")
print ("-----------------")
"""Conjugate Gradient"""
reg_ConGrad = fc.ConjugateGrad(y_tr_norm, x_tr_norm, y_test_values, x_test_norm, y_tr_mean , y_tr_var)
reg_ConGrad.run()
reg_ConGrad.print_result("'''Conjugate Gradient Algorithm'''")
reg_ConGrad.plot_w(features,"Conjugate Gradient optimal soution vector: ")
reg_ConGrad.plot_estimation("Conjugate Gradient: estimated solution vs real solution")
reg_ConGrad.plotError("CONJUGATE GRADIENT: mean square error")
reg_ConGrad.plot_Histograms("Conjugate Gradient: histograms of the error ")
reg_ConGrad.mean_std_train("Conjugate Gradient mean and standard deviation of regression errors for training dataset: ")
reg_ConGrad.mean_std_test("Conjugate Gradient mean and standard deviation of regression errors for test dataset: ")
reg_ConGrad.MSETraining("Conjugate Gradient MSE of the training set: ")
reg_ConGrad.MSETest("Conjugate Gradient MSE of the test set: ")
reg_ConGrad.r_squared("Coefficiente di determinazione R^2")
print ("-----------------")

print("STOCHASTIC GRADIENT USING ADAMS:")
reg_Stoca = fc.Stochastic(y_tr_norm, x_tr_norm, y_test_values, x_test_norm, y_tr_mean , y_tr_var)
reg_Stoca.run(x_val_norm, y_val_norm)
reg_Stoca.print_result("'''Stochastic Graidient'''")
reg_Stoca.plot_w(features,"Stochastic Graidient optimal soution vector: ")
reg_Stoca.plot_estimation("Stochastic Graidient estimated solution vs real solution:")
reg_Stoca.plot_Histograms_it("Stochastic Graidient: histograms of the estimation error: ", x_val_norm, y_val)
reg_Stoca.mean_std_train("Stochastic Graidient: mean and standard deviation of regression errors for training dataset: ")
reg_Stoca.mean_std_test("Stochastic Graidient: mean and standard deviation of regression errors for test dataset: ")
reg_Stoca.mean_std_val("Stochastic Graidient: mean and standard deviation of regression errors for validation dataset: ", y_val, x_val_norm)
reg_Stoca.MSETraining("Stochastic Graidient MSE of the training set: ")
reg_Stoca.MSEValidation("Stochastic Gradient MSE of the validation set: ",x_val_norm, y_val)
reg_Stoca.MSETest("Stochastic Graidient MSE of the test set: ")
reg_Stoca.r_squared("Coefficiente di determinazione R^2")
print ("-----------------")



"""Ridge regression"""
print("RIDGE REGRESSION")
reg_Ridge = fc.RidgeRegression(y_tr_norm, x_tr_norm, y_test_values, x_test_norm, y_tr_mean , y_tr_var)
reg_Ridge.run(x_val_norm, y_val_norm)
reg_Ridge.print_result("-")
reg_Ridge.plot_w(features,"Ridge regression optimal solution vector: ")
reg_Ridge.plot_err_lambda()
reg_Ridge.plot_estimation("Ridge regression estimated solution vs real solution: ")
reg_Ridge.plot_Histograms_it("Ridge regression histograms of the estimation error: ",x_val_norm, y_val)
reg_Ridge.mean_std_train("Ridge regression: mean and standard deviation of regression errors for training dataset: ")
reg_Ridge.mean_std_test("Ridge regression: mean and standard deviation of regression errors for test dataset: ")
reg_Ridge.mean_std_val("Ridge regression: mean and standard deviation of regression errors for validation dataset: ", y_val, x_val_norm)
reg_Ridge.MSETraining("Ridge Regression MSE of the training set: ")
reg_Ridge.MSEValidation("Ridge Regression MSE of the validation set: ",x_val_norm, y_val)
reg_Ridge.MSETest("Ridge Regression MSE of the test set: ")
reg_Ridge.r_squared("Coefficiente di determinazione R^2")

