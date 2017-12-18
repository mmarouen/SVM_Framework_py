#gateway function for the solver
#output model parameters + prediction

def RKHS(Input,Y,Xtest=None,yTest=None, #input data
         opType='Classification', #classification or regression
         classifier='SVM', #classifier name 'SVM', 'LS', 'Softmax'
         kernel='linear', #kernel type: 'linear', 'gaussian', 'polynomial'
         degree=3, #polynom degree
         gamm=1, #RBF parameter
         C=1, #cost parameter: lambda=1/(2*C)
         learning_rate=None, 
         tol=1e-2,
         epochs=400, #number of iterations
         traceObj=False, 
         gradCheck=False,
         optMode='NGD' #optimization method 'NGD', 'CGD'
        ):
    import numpy as np
    rsp=transformResponse(Y,classifier,opType)
    resp=rsp['respMat']
    cl=rsp['Classes']
    X=np.asmatrix(Input)
    yhat=np.array([])
    betahat=np.array([])
    sv=np.array([])
    lambd=1/(2*C)
    
    KerMat=buildKernel(kernel,X,X,degree=degree,gamm=gamm)
    Ker=KerMat['KerMat']
    H=KerMat['H']
    S=KerMat['S']
    
    if (opType=='Regression') & (classifier=='LS'):
        inv=np.linalg.inv(np.dot(H.T,H)+lambd*S)
        betahat=np.dot(np.dot(inv,H.T),resp)
    
    if opType=='Classification':
        if optMode=='NGD': out=NGD(X,resp,Ker,classifier,lambd,tol,epochs)
        #if optMode=='CGD': out=CGD(X,resp,XTest,resp1,Ker,kerT,H,S,classifier,lambd,tol,epochs)
        betahat=out['betahat']
        n_iter=out['n_iter']
        sv=out['sv']
        
    f_x=np.dot(H,betahat).T
    out=transformOutput(f_x,cl,opType)
    yhat=out['yhat']
    modelargs={'kername':kernel,'gamm':gamm,'degree':degree,'X':X,'y':resp,'epochs':epochs,
                 'lambd':lambd,'classes':cl,'opType':opType}
    return{'yhat':yhat,'fx':f_x,'yhatMat':out['yhatMat'],'beta':betahat,'sv':sv,'rkhsargs':modelargs,
           'n_iter':n_iter}
