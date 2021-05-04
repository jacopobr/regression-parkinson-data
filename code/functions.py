"""
@author: Jacopo Braccio
"""
import numpy as np
import matplotlib.pyplot as plt

class SolveMinProbl:
    def __init__(self, y , X, y_test, X_test , mean, var): #inizialization
        self.matr = X
        self.ns = y.shape[0] #number of rows 
        self.nf = X.shape[1] #number of columns
        self.vect = y #column vector y
        self.xtest = X_test #test dataset
        self.ytest = y_test #total UPDRS of test dataset
        self.mean = mean #mean of the training set
        self.var = var #std of the traning set
        self.sol = np.zeros((self.nf,1),dtype = float) #column vector w
        
    #print the optimal vector for *title* algorithm 
    def print_result(self,title): #method to print 
        print(title, "_:")
        print("The_optimun_weight_vector_is:_")
        print(self.sol)
        return
    
    #Plotting of the solution
    def plot_w (self,features, title = "Solution"): #method to plot
        w = self.sol 
        n = np.arange (self.nf)
        plt.figure()
        plt.bar(n,w)
        plt.xticks(n,features, rotation = 'vertical')
        plt.xlabel("feature n")
        plt.ylabel("w(n)")
        plt.xticks(np.arange(0,19,1))
        plt.title(title)
        plt.grid()
        plt.show()
        return
    
    #print of the MEAN SQUARE ERROR of the training dataset
    def MSETraining(self, title):
        y_train_estimated = self.var * np.dot(self.matr,self.sol) + self.mean
        y_train = self.var * self.vect + self.mean
        MSE = (np.linalg.norm(y_train - y_train_estimated)**2)/self.matr.shape[0]
        print(title)
        print(round(MSE,3))
        self.msetr = MSE
        return
    
    #print of the MEAN SQUARE ERROR of the test dataset
    def MSETest(self,title):
        y_test_estimated = self.var * np.dot(self.xtest,self.sol) + self.mean
        MSE = (np.linalg.norm(self.ytest - y_test_estimated)**2)/self.xtest.shape[0]
        print(title)
        print(round(MSE,3))
        return
    #print of the MEAN SQUARE ERROR of the validation dataset
    def MSEValidation(self, title,x_val,y_val):
        y_val_estimated = self.var * np.dot(x_val,self.sol) + self.mean
        MSE = (np.linalg.norm(y_val - y_val_estimated)**2)/x_val.shape[0]
        print(title)
        print(round(MSE,3))
        self.msetr = MSE
        return
    #scatter plot of y_estimated and y_real
    def plot_estimation(self, title): 
        y_estimated = self.var * np.dot(self.xtest,self.sol) + self.mean
        plt.scatter(y_estimated,self.ytest,)
        plt.plot([5,55], [5, 55], color='r')
        plt.xlabel('Y Estimated')
        plt.ylabel('Y real')
        plt.title(title)
        plt.grid()
        plt.show()
        return 
    
    #plot histograms of the erros for training and test set
    def plot_Histograms(self, title):
        y_tr_un = self.var * self.vect + self.mean
        y_estimated_tr = self.var * np.dot(self.matr,self.sol) + self.mean
        e_train = (y_tr_un - y_estimated_tr)
        y_test_estimated = self.var * np.dot(self.xtest,self.sol) + self.mean
        e_test = self.ytest - y_test_estimated
        plt.hist(e_test,bins = 20, density = True, label = "Test Error pdf", alpha = 0.7, histtype='bar')
        plt.hist(e_train, bins = 20, density = True, label = "Training Error pdf",alpha = 0.7, histtype='bar')
        plt.legend( loc = 'best')
        plt.xlabel('e')
        plt.ylabel('Frequency ')
        plt.title(title)
        plt.show()
        self.e_train = e_train
        return 
    #plot histograms of the erros for training, test and validation set
    def plot_Histograms_it(self, title, x_val, y_val):
        y_estimated_val= self.var * np.dot(x_val,self.sol) + self.mean
        e_validation = (y_val - y_estimated_val)
        y_tr_un = self.var * self.vect + self.mean
        y_estimated_tr = self.var * np.dot(self.matr,self.sol) + self.mean
        e_train = (y_tr_un - y_estimated_tr)
        y_test_estimated = self.var * np.dot(self.xtest,self.sol) + self.mean
        e_test = self.ytest - y_test_estimated
        plt.hist(e_test,bins = 20, density = True, label = "Test Error pdf", alpha = 0.7, histtype='bar')
        plt.hist(e_train, bins = 20, density = True, label = "Training Error pdf",alpha = 0.7, histtype='bar')
        plt.hist(e_validation,bins = 20, density = True, label = "Test Error pdf", alpha = 0.7, histtype='bar')
        plt.legend( loc = 'best')
        plt.xlabel('e')
        plt.ylabel('Frequency ')
        plt.title(title)
        plt.show()
        return 
    #print the coefficient of determinatio R^2
    def r_squared(self,title):
        RMSE = np.sqrt(self.msetr)
        r_2 = 1 - ((RMSE**2)/self.var**2)
        print(title)
        print(round(r_2,3))
   #print the mean and std of training regression errors 
    def mean_std_train(self,title):
        print(title)
        y_tr = self.var * self.vect + self.mean
        err = (y_tr - (self.var * np.dot(self.matr,self.sol) + self.mean))
        print (round(err.mean(),3))
        print(round(err.std(),3))
        return
    #print the mean and std of test regression errors 
    def mean_std_test(self,title):
        print(title)
        y_test_estimated = self.var * np.dot(self.xtest,self.sol) + self.mean
        err = self.ytest -y_test_estimated
        print(round(err.mean(),3))
        print(round(err.std(),3))
        return
    #print the mean and std of validation regression errors 
    def mean_std_val(self,title, y_val,x_val):
        print(title)
        y_val_estimated = self.var * np.dot(x_val,self.sol) + self.mean
        err = y_val - y_val_estimated
        print(round(err.mean(),3))
        print(round(err.std(),3))
        return
        
        
    
    
"""SolveLLS class used to solve the regression problem with Linear Least Square Algorithm"""
class SolveLLS (SolveMinProbl):
    def run(self):
        X = self.matr
        y = self.vect
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y) #solution vector
        self.sol = w
        return (w)
     
""" Solve ConjugateGrad class used to solve the regression problem with 
conjugate gradient algorithm. A description of the algorithm is given in the report"""
class ConjugateGrad(SolveMinProbl):
    
    def run(self):
        X = self.matr
        y = self.vect
        nf = X.shape[1] #number of columns
        Q = np.dot(X.T,X) 
        b = np.dot(X.T,y)
        w = np.zeros((self.nf,1),dtype = float)
        d = b
        g = -b
        self.err = np.zeros ((self.nf, 2), dtype=float)
        
        for k in range (nf):
            alpha = - (np.dot(d.T,g))/(np.linalg.multi_dot([d.T,Q,d]))
            w = w + alpha * d
            g = g + alpha * np.dot(Q,d)
            beta = (np.linalg.multi_dot([g.T,Q,d]))/(np.linalg.multi_dot([d.T,Q,d]))
            d = - g + beta * d
            self.err[k,0] = k
            self.err[k,1] = np.square((self.var * y + self.mean) - (self.var * np.dot(X,w[0,:]) + self.mean)).mean()
        w = (w[0,:])
        self.sol = w
        return 

    #function used to plot the error as funciont of iterations
    def plotError(self, title, logy = 0, logx = 0):
        err = self.err
        plt.figure ()
        if (logy == 0) and (logx == 0):
            plt.plot(err[:, 0], err[:, 1], label='Training dataset')
        if (logy == 1) and (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1], label='Training dataset')
        if (logy == 0) and (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1], label='Training dataset')
        if (logy == 1) and (logx == 1):
            plt.loglog(err[:, 0], err[:, 1], label='Training dataset')
        plt.xlabel (' feature n')
        plt.ylabel ('e(n)')
        plt.xticks(np.arange(0,19,1))
        plt.title (title)
        plt.margins (0.01, 0.1)
        plt.grid ()
        plt.legend(loc='upper right')

        plt.show ()
        return

"""Class Stochastic used to solve the regression problem with the Stochastic gradient using ADAM optimizer."""
class Stochastic(SolveMinProbl):
    
    def run(self, x_val, y_val, eps = 0.1):
        
        X = self.matr #training set
        y = self.vect #total UPDRS for training set
        w = np.random.rand(X.shape[1],)
        gamma = 0.01 #learning coefficient
        beta_1 =  0.9#used for the estimation of the mean 
        beta_2 = 0.999 #used for the estimation of the variance 
        epsilon = 1e-8 #correction coefficent
        estimean = 0
        estivar = 0
        Nit = 20000
        MSE = np.zeros((Nit, 1), dtype=float)
        self.err = np.zeros((Nit,2), dtype = float)
        self.err_val = np.zeros((Nit,2), dtype = float)
        solution = np.zeros((Nit,19), dtype = float) #Auxiliary matrix used to store the solution, so that the solution at i-50 can be recalled
        
        stop = 0 #counter for stopping condition
        
        for n in range (Nit):
            if stop < 50:
                 m = np.mod(n,X.shape[0])
                 x = X[m,:]
                 err_0 = np.dot(x,w)-y[m]
                 grad = 2 * err_0 * x
                 estimean = beta_1*estimean + (1-beta_1)*grad #calculate the avarage mean
                 estivar = beta_2*estivar + (1-beta_2)*(grad**2) #calculate the avarage variance
                 estimean_cor = estimean/(1-beta_1**(n+1))  #correct using the correction coefficient
                 estivar_cor = estivar/(1-beta_2**(n+1)) #correct using the correction coefficient
                 if np.abs(err_0) < 0.1:
                    grad = grad * 0
                 w = w - (gamma*estimean_cor)/(((estivar_cor) + epsilon)**(1/2))
                 solution[n] = w
                 prodotto = np.dot(x_val,w)
                 num_mse = np.linalg.norm(y_val - prodotto)**2
                 MSE[m] = num_mse / x_val.shape[0]
                 self.err_val[n,0] = n
                 self.err_val[n,1] = MSE[m]
                 self.err[n,0] = n
                 self.err[n,1] = np.square(np.linalg.norm(self.vect - np.dot(self.matr,w)))/self.matr.shape[0]
                 
                 #if the MSE at iteration m-1 is bigger than the MSE at m-1 the counter increases
                 if MSE[m] > MSE[m-1]:
                     stop +=1
                 else:
                    stop = 0 #since the 50 increases must be consecutive, the stop condition is rest if the previous if is faulse.
            else: 
                print("algorithm stopped at: ",n)
                break
        
        #print the mse for validation and training set as funcion of iterations   
        err_V= self.err_val[0:n]
        err = self.err[0:n]
        self.sol = solution[n-50]
        logy=0
        logx=0
        plt.figure ()
        if (logy == 0) and (logx == 0):
            plt.plot(err[:, 0], err[:, 1], label='Training dataset')
            plt.plot(err_V[:, 0], err_V[:, 1],'r', label='Validation dataset')
        if (logy == 1) and (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1], label='Training dataset')
            plt.semilogy(err_V[:, 0], err_V[:, 1],'r', label='Validation dataset')
        if (logy == 0) and (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1], label='Training dataset')
            plt.semilogx(err_V[:, 0], err_V[:, 1],'r', label='Validation dataset')
        if (logy == 1) and (logx == 1):
            plt.loglog(err[:, 0], err[:, 1], label='Training dataset')
            plt.loglog(err_V[:, 0], err_V[:, 1],'r', label='Validation dataset')
        plt.xlabel ('n')
        plt.ylabel ('e(n)')
        plt.title ("Error(iteration)")
        plt.margins (0.01, 0.1)
        plt.grid ()
        plt.legend(loc='best')
        plt.show ()
        
        return ()
"""RidgeRegression used to solve the regression problem with the Stochastic gradient using ridge regression algorithm."""
class RidgeRegression(SolveMinProbl):
    
    def run(self, x_val, y_val):
        
        X = self.matr
        y = self.vect
        I = np.eye(self.nf) #idenity matrix
        setlambda = np.arange(0, 50, 0.1) #set of possible values of lambda
        numbers_lambda = len(setlambda) #counts the number of element inside the set of lambda chosen
        self.err = np.zeros((numbers_lambda,2))
        self.err_val = np.zeros((numbers_lambda,2))
        n = 0 #used for the training error
        for lambdait in setlambda:
            lam = lambdait ** 10
            w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+ lam * I),X.T),y) 
            self.err[n,0] = lam
            self.err[n,1] = np.square(np.linalg.norm(np.dot(X,w)-y))/X.shape[0] #MSE of the training set 
            self.err_val[n,0] = lam
            self.err_val[n,1] = np.square(np.linalg.norm(y_val - np.dot(x_val,w)))/x_val.shape[0]
            n = n + 1
            
        index_min_lambda = np.argmin(self.err_val[:,1]) #return the index of the minumum element in the array of validation error
        lambda_opt = self.err_val[index_min_lambda,0] #the lambda corresponding the minumum index is chosen as optimal
        print ("Lambda optimum is : ",lambda_opt)
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+ lambda_opt * I),X.T),y) #final solution
        self.sol = w
        return
    
    #function that prints the errors of the validation and training dataset as function of lambda
    def plot_err_lambda(self):
        
        err = self.err
        err_val = self.err_val
        logy=0
        logx=0
        plt.figure ()
        
        if (logy == 0) and (logx == 0):
            plt.plot(np.log(err[:, 0]), err[:, 1], label='Training Error')
            plt.plot(np.log(err_val[:,0]), err_val[:,1], label = 'Validation Error')
        if (logy == 1) and (logx == 0):
            plt.plot(np.log(err[:, 0]), err[:, 1], label='Training dataset')
            plt.plot(np.log(err_val[:,0]), err_val[:,1], label = 'Validation Error')
        if (logy == 0) and (logx == 1):
            plt.plot(np.log(err[:, 0]), err[:, 1], label='Training dataset')
            plt.plot(np.log(err_val[:,0]), err_val[:,1], label = 'Validation Error')
        if (logy == 1) and (logx == 1):
            plt.plot(np.log(err[:, 0]), err[:, 1], label='Training dataset')
            plt.plot(np.log(err_val[:,0]), err_val[:,1], label = 'Validation Error')
        plt.xlabel ('lambda')
        plt.ylabel ('e(lambda)')
        plt.title ("Variation of the error as function of lambda'")
        plt.legend( loc = 'best')
        plt.xlim((0,40))
        plt.grid ()
        plt.show ()
        return
        
    
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
