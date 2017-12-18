#converts the response vector into adequate matrix form
def transformResponse(response,classifier,tt='Regression'):
    import numpy as np
    N=len(response)
    response=np.reshape(response,(N,1))
    respMat=np.array([])
    classes=np.array([])
    if (tt=='Regression'): respMat=np.asmatrix(response)
    if (tt=='Classification'):
        K=len(np.unique(response))
        classes=np.unique(response)
        if (K==2) & (classifier=='Softmax'): respMat=np.asmatrix(response)
        if (K==2) & (classifier=='SVM'):
            if np.any(response==0): response=2*(response-0.5)
            respMat=np.asmatrix(response)
            
    return {'Response':response,'respMat':respMat,'Classes':classes}
    
#converts algorithm output into vector form similar to initial response
def transformOutput(f_x,CL,tt='Regression'):
    import numpy as np
    N,K=f_x.shape
    yhat=np.array([])
    if tt=='Regression': yhat=f_x
    if tt=='Classification':
        if K==1:
            fx=np.squeeze(np.asarray(f_x))
            yhat=np.repeat(min(CL),N)
            yhat[fx>0]=max(CL)
        if K>2: yhat=np.apply_along_axis(lambda x:CL[np.argmax(x)],1,f_x)
        yhat=yhat.reshape((N,1))
    return {'yhat':yhat,'yhatMat':f_x}

def Softmax(X):
    import numpy as np
    X=np.matrix(X)
    eps=1e-15
    Eps=1-eps
    M=np.max(X)
    prod=np.apply_along_axis(lambda x: np.exp(x-M)/np.sum(np.exp(X-M),axis=1),0,X)
    prod[prod>Eps]=Eps
    prod[prod<eps]=eps
    return(prod)

def buildKernel(ker,X,Z,degree=3,gamm=0.1,pred=False):
    import numpy as np
    H=np.matrix([])
    S=np.matrix([])
    if ker=='linear': degree=1
    if ker in ['linear','polynomial']: kermat=np.power((np.dot(Z,X.T)+1),degree)-1
    if ker=='gaussian': kermat=np.apply_along_axis(lambda x:np.exp(-gamm*np.sum(np.power(Z-x,2),1)).T,1,X)
    if(pred is False):
        N,_=kermat.shape
        H=np.hstack((np.ones(N).reshape(N,1),kermat))
        S=np.vstack((np.zeros(N),kermat))
        S=np.hstack((np.zeros(N+1).reshape(N+1,1),S))
    return {'H':H,'KerMat':kermat,'S':S}



