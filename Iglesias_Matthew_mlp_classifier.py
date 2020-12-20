#Matthew Iglesias

import numpy as np
from utils import *
import math 
import time
from sklearn.neural_network import MLPClassifier
    
if __name__ == "__main__":  
    plt.close('all')
    
    #data_path = 'C:\\Users\\OFuentes\\Documents\\Research\\data\\'  # Use your own path here
    
    X = np.load('mnist_X.npy').reshape(-1,28*28)/255
    y = np.load('mnist_y.npy')
        
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    model = MLPClassifier(solver='adam', alpha=1e-6, hidden_layer_sizes=(500),activation='relu', verbose=True, random_state=1,max_iter=10)
    
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    print('Training iterations  {} '.format(model.n_iter_))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy: {0:.6f}'.format(accuracy(y_test,pred)))
          
    cm = confusion_matrix(y_test,pred)   
    print(cm)
    
   