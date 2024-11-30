import numpy as np
import math

LightSpeed=299792.458

def Vp_to_logd(vpec,Rsf,OmegaM):
    deccel = 3.0*OmegaM/2.0 - 1.0
    Vmod   = Rsf*LightSpeed*(1.0 + 0.5*(1.0 - deccel)*Rsf - (2.0 - deccel - 3.0*deccel*deccel)*Rsf*Rsf/6.0)
    Logd   =    vpec / ( math.log(10.)*Vmod/(1.0+Vmod/LightSpeed)  )
    return Logd
def logd_to_Vp(Logd,Rsf,OmegaM):
    deccel = 3.0*OmegaM/2.0 - 1.0
    Vmod   = Rsf*LightSpeed*(1.0 + 0.5*(1.0 - deccel)*Rsf - (2.0 - deccel - 3.0*deccel*deccel)*Rsf*Rsf/6.0)
    vpec   = math.log(10.)*Vmod/(1.0+Vmod/LightSpeed) * Logd
    return vpec
def VpRand_Fun(logd,cz,czR,NczBin,OmegaM):
    pv=logd_to_Vp(logd,cz/LightSpeed,OmegaM)
    pvR=[];ev=[]
    Num,Bin=np.histogram(cz,NczBin);
    for i in range(len(Num)+1):
        if(i<=len(Num)-1):
            ind=(cz>=Bin[i])&(cz<=Bin[i+1])
            er=np.std(pv[ind])
            ev.append(er)
    ev=np.array(ev)
    x=Bin[:-1]+0.5*np.diff(Bin)
    k,b=np.polyfit(x,ev,deg=1)
    epvR=k*czR+b
    np.random.seed(326)
    pvR=np.random.normal(loc=0.0, scale=epvR, size=len(epvR))
    return pvR, epvR

    

def FormatData_Fun(Datas,Output_dir,Datype):
    if(Datype=='gal-survey')or(Datype=='gal-mock'):
        Ra,Dec,cz,nb,FKPden=Datas    
        w=1.*np.ones(len(Ra))
        wfkpden =1./(   1.+nb*FKPden)
        ndata=len(np.where(nb>0)[0])
        outfile = open(Output_dir, 'w')
        outfile.write("#   %18d\n"% (ndata))
        for i in range(len(Ra)):
          if(nb[i]>0):  
            outfile.write(" %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf \n" % (Ra[i],Dec[i],cz[i],nb[i],w[i]*wfkpden[i] ))  
        outfile.close()
        wfkp= wfkpden 
    if(Datype=='gal-rand'):
        Ra,Dec,cz,nb,FKPden =Datas
        w=1.*np.ones(len(Ra))
        wfkpden  = 1./(   1.+nb*FKPden)
        ndata=len(np.where(nb>0)[0]) 
        outfile = open(Output_dir, 'w')
        outfile.write("#   %18d\n"% (ndata))
        for i in range(len(Ra)):
          if(nb[i]>0):  
            outfile.write(" %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf \n" % (Ra[i],Dec[i],cz[i], nb[i],w[i]*wfkpden[i] ))  
        outfile.close()    
        wfkp= wfkpden 
    if(Datype=='pv-survey')or(Datype=='pv-mock'):
        Ra,Dec,cz,logd,elogd,nb,sig_v ,FKPden, FKPmom=Datas    
        w=1.*np.ones(len(Ra))
        wfkpmom =1./(   (cz/(1.+cz/299792.458)*np.log(10.)*elogd)**2+sig_v**2 + nb*FKPmom   )
        wfkpden =1./(   1.+nb*FKPden)
        ndata=len(np.where(nb>0)[0])
        outfile = open(Output_dir, 'w')
        outfile.write("#   %18d\n"% (ndata))
        for i in range(len(Ra)):
          if(nb[i]>0):  
            outfile.write(" %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %21.18lf  %12.6lf \n" % (Ra[i],Dec[i],cz[i],logd[i],nb[i],w[i]*wfkpmom[i],w[i]*wfkpden[i] ))  
        outfile.close()
        wfkp=[wfkpden,wfkpmom]
    if(Datype=='pv-rand'):
        Ra,Dec,cz,pvr,epv,nb,sig_v,FKPden, FKPmom=Datas
        w=1.*np.ones(len(Ra))
        wfkpmom  = 1./(epv**2+sig_v**2+ nb*FKPmom) 
        wfkpden =1./(   1.+nb*FKPden)
        ndata=len(np.where(nb>0)[0]) 
        outfile = open(Output_dir, 'w')
        outfile.write("#   %18d\n"% (ndata))
        for i in range(len(Ra)):
          if(nb[i]>0):  
            outfile.write(" %12.6lf  %12.6lf  %12.6lf  %12.6lf  %12.6lf  %21.18lf  %12.6lf \n" % (Ra[i],Dec[i],cz[i], pvr[i], nb[i],w[i]*wfkpmom[i],w[i]*wfkpden[i] ))  
        outfile.close() 
        wfkp=[wfkpden,wfkpmom]
    return Output_dir,wfkp

