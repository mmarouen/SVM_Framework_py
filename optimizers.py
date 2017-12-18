#performs newton-raphson optimization
def NGD(X,resp,Ker,classifier,lambd,tol,epochs):
    import numpy as np
    N,K=resp.shape
    sv=np.repeat(True,N)
    betahat=np.array(np.repeat(0,N+1),dtype=float)
    intercept=0
    t=0
    if (classifier=="SVM") & (K==1) :
        sv_new=np.random.choice([True,False],N,replace=True)
        while (t<10) & np.any(sv_new != sv):
            t+=1
            if t>1: sv=sv_new
            betahat=np.array(np.repeat(0,N),dtype=float)
            intercept=0
            Ksv=Ker[sv,:]
            Ksv=Ksv[:,sv]
            Ysv=np.insert(resp[sv],0,0).T
            N1=sum(sv)
            IntMat=Ksv+lambd*np.diag(np.ones(N1))
            IntMat=np.hstack((np.ones(N1).reshape(N1,1),IntMat))
            IntMat=np.vstack((np.ones(N1+1).reshape(1,N1+1),IntMat))
            IntMat[0,0]=0
            bsv=np.dot(np.linalg.inv(IntMat),Ysv)
            intercept=bsv[0]
            betahat[sv]=bsv[1:]
            yhat=np.dot(Ker,betahat)+intercept
            yhat=np.squeeze(np.asarray(yhat))
            sv_new=np.squeeze(np.asarray(resp))*yhat<1
            betahat=np.insert(betahat,0,intercept)
            
    return {'betahat':betahat,'n_iter':t,'sv':sv_new}
