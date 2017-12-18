# SVM_Framework
Support vector machines flexible framework  
We solve the unconstrained primal SVM formulation    
SVM & Softmax classifiers supported  
NB:  
-Softmax classifier refers to penalized and kernalized logistic regression  
-Classical logistic regression can be obtained by setting cost very high & using a linear kernel  
-R implementation in a seperate repository  
# 1. Classifiers/regressors:  
LS: regression classifier using penalized least squared loss  
Softmax: Softmax classifier using cross entropy loss  
SVM:svm classifier using quadratic hinge loss  
# 2. Optimization methods:  
BGD: gradient descent (batch)  
NGD: Newton-Raphson optimization (batch)  
CGD: conjugate gradient descent (batch)    
SGD: stochastic gradient descent (under development)  
# 3. Kernels:  
gaussian: gaussian kernel  
linear: linear kernel  
poly:polynomial kernel  
For any remarks please let me know <azzouz.marouen@gmail.com>
