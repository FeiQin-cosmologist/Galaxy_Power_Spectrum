import numpy as np
from scipy import stats
import emcee
from PSmodFun import *
import matplotlib.pyplot as plt
import scipy as sp


Sig_ZP =3.*10**5#np.nan # fake zero point.



###############################################################################
######                                                                   ######
######                        Sec 1. Tool Functions                      ######
######                                                                   ######
############################################################################### 
# 1. Loading measured PS and convmat//l
def LoadPS_Fun(Input_dir,Nmock,PS_type,WFCon_type,OnlyL0,PS_measure ): 
    if(PS_measure =='mockAve')or(PS_measure =='mockML'):PS_measure='mocks'
    k_data=[];Pk_data=[]
    nl=3;nlc=2
    if( WFCon_type == 'Ross')or(OnlyL0): nl=nlc=1
    if('den' in PS_type ):
        infile=np.loadtxt(Input_dir+'Data_den')
        Nk=len(infile[0:len(infile[:,0])-1,1])
        for i in range(nl):
            k_data.append(infile[0:len(infile[:,0])-1,1])
            Pk_data.append(infile[0:len(infile[:,0])-1,2*(i+1)])
    if('mom' in PS_type ):
        infile=np.loadtxt(Input_dir+'Data_mom')
        Nk=len(infile[0:len(infile[:,0])-1,1])
        for i in range(nl):
            k_data.append(infile[0:len(infile[:,0])-1,1])
            Pk_data.append(infile[0:len(infile[:,0])-1,2*(i+1)])
    if('crs' in PS_type ):    
        infile=np.loadtxt(Input_dir+'Data_crs')
        Nk=len(infile[0:len(infile[:,0])-1,1])
        for i in range(nlc):
            k_data.append(infile[0:len(infile[:,0])-1,1])
            Pk_data.append(infile[0:len(infile[:,0])-1,2*(i+1)+1])
    k_data=np.concatenate(k_data)
    Pk_data=np.concatenate(Pk_data)
    # 2: PS of mocks:-----------------------------------  
    k_mock=[];  Pk_mockap=[]
    for i_mock in range(Nmock):
        Pk_mockl=[] ;
        if('den' in PS_type ):
            infile=np.loadtxt(Input_dir+'Mock_den'+str(i_mock))
            for i in range(nl):
                if(i_mock==0): k_mock.append(infile[0:len(infile[:,0])-1,1])
                Pk_mockl.append(infile[0:len(infile[:,0])-1,2*(i+1)])
        if('mom' in PS_type ):
            infile=np.loadtxt(Input_dir+'Mock_mom'+str(i_mock))
            for i in range(nl):
                if(i_mock==0): k_mock.append(infile[0:len(infile[:,0])-1,1])
                Pk_mockl.append(infile[0:len(infile[:,0])-1,2*(i+1)])    
        if('crs' in PS_type ):
            infile=np.loadtxt(Input_dir+'Mock_crs'+str(i_mock))
            for i in range(nlc):
                if(i_mock==0): k_mock.append(infile[0:len(infile[:,0])-1,1])
                Pk_mockl.append(infile[0:len(infile[:,0])-1,2*(i+1)+1])
        Pk_mockap.append(np.concatenate(Pk_mockl))
    k_mock=np.concatenate(k_mock)
    Pk_mock=np.zeros((np.shape(Pk_mockap)[1],np.shape(Pk_mockap)[0]))
    for i in range(Nmock):
        Pk_mock[:,i]=Pk_mockap[i]
    # conv mat:------------------------------------------- 
    w_rand = np.nan
    if( WFCon_type != 'Ross'):
        PS_type='den-024 mom-024 crs-13'
        conv=[];Kdj=[];Kpj=[];Kcj=[]
        if('den' in PS_type ):
            covden,ki,kdj=np.load(Input_dir+'convden_'+WFCon_type+'_'+PS_measure+'.npy',allow_pickle=True) 
            conv.append(covden);Kdj.append(kdj)
        if('mom' in PS_type ):
            covmom,ki,kpj=np.load(Input_dir+'convmom_'+WFCon_type+'_'+PS_measure+'.npy',allow_pickle=True)
            conv.append(covmom) ;Kpj.append(kpj)
        if('crs' in PS_type ): 
            covcrs,ki,kcj=np.load(Input_dir+'convcrs_'+WFCon_type+'_'+PS_measure+'.npy',allow_pickle=True)
            conv.append(covcrs);Kcj.append(kcj)
        kcon=[np.concatenate(Kdj),np.concatenate(Kpj),np.concatenate(Kcj)]    
    if( WFCon_type == 'Ross'):
        if('den' in PS_type ):
            conv,kir,kjr, w_rand=np.load(Input_dir+'convden_'+WFCon_type+'_'+PS_measure+'.npy',allow_pickle=True) 
        if('mom' in PS_type ):
            conv,kir,kjr, w_rand=np.load(Input_dir+'convmom_'+WFCon_type+'_'+PS_measure+'.npy',allow_pickle=True) 
        kcon=kjr
    return Nk,k_data, Pk_data, k_mock,Pk_mock,conv , kcon , w_rand 
# 2. MLE or averge of mocks:---------------------------------------------------
def PSmock_prop(k_mock,Pk_mock,fit_minK ,fit_maxK,fit_tech):
    Pave=np.zeros(len(k_mock));
    ePk_obs=np.zeros(len(k_mock));
    ekPk_obs=np.zeros(len(k_mock));
    for i in range(len(k_mock)):
        Pave[i]=np.mean(Pk_mock[i,:])
        ePk_obs[i]=np.std(Pk_mock[i,:])
        ekPk_obs[i]=np.std(k_mock[i]*Pk_mock[i,:])
    if(fit_tech=='mockML'):
        Pave=np.zeros(len(k_mock));
        for i in range(len(k_mock)):
          if(k_mock[i]>=fit_minK)and(k_mock[i]<=fit_maxK):  
            Gaussian_KDE = stats.gaussian_kde(Pk_mock[i,:])        
            ss=Pk_mock[i,Gaussian_KDE.pdf(Pk_mock[i,:])==max(Gaussian_KDE.pdf(Pk_mock[i,:]))  ]
            Pave[i]=ss[0]
    return Pave,   ePk_obs,  ekPk_obs 
################          The end of Sec.1          ###########################
###############################################################################



 
















###############################################################################
######                                                                   ######
######                     Sec 2. Fitting Functions                      ######
######                                                                   ######
###############################################################################   
def ParmPSmod_Fun(parm,ps_type):
    if(ps_type=='den-024 mom-024')or(ps_type=='den-024 mom-02')or(ps_type=='den-02 mom-02')or(ps_type=='den-024 mom-024 crs-13')or(ps_type=='den-02 mom-02 crs-1'):
        fsigma8,  b1sigma8,b2sigma8,b3nlsigma8,  b1vsigma8,   sigmav1_squre,sigmav2_squre,sigmavv1_squre,sigmavv2_squre = parm     
        # den ------------------- 
        bssigma8 ='anal'
        sigmav1_squre=np.abs(sigmav1_squre)
        sigmav2_squre=np.abs(sigmav2_squre)
        sigmav3_squre=sigmav1_squre
        # mom ------------------- 
        b2vsigma8=b2sigma8
        bsvsigma8 ='anal'
        b3nlvsigma8='anal'
        sigmavv1_squre=np.abs(sigmavv1_squre)
        sigmavv2_squre=np.abs(sigmavv2_squre)
        sigmavv3_squre=sigmavv1_squre  
    if(ps_type=='den-02 mom-0')or(ps_type=='den-024 mom-0')or(ps_type=='den-0 mom-0 crs-1'):
        fsigma8,  b1sigma8,b2sigma8,b3nlsigma8,  b1vsigma8,   sigmav1_squre,sigmav2_squre,sigmavv_squre = parm     
        # den ------------------- 
        bssigma8 ='anal'
        sigmav1_squre=np.abs(sigmav1_squre)
        sigmav2_squre=np.abs(sigmav2_squre)
        sigmav3_squre=sigmav1_squre
        # mom ------------------- 
        b2vsigma8=b2sigma8
        bsvsigma8 ='anal'
        b3nlvsigma8='anal'
        sigmavv1_squre=np.abs(sigmavv_squre)
        sigmavv2_squre=sigmavv1_squre
        sigmavv3_squre=sigmavv1_squre       
    if(ps_type=='den-0 mom-0'):
        fsigma8,  b1sigma8,b2sigma8,             b1vsigma8,   sigmav1_squre,sigmav2_squre,sigmavv_squre = parm     
        # den ------------------- 
        b3nlsigma8='anal'
        bssigma8 ='anal'
        sigmav1_squre=np.abs(sigmav1_squre)
        sigmav2_squre=np.abs(sigmav2_squre)
        sigmav3_squre=sigmav1_squre
        # mom ------------------- 
        b2vsigma8=b2sigma8
        bsvsigma8 ='anal'
        b3nlvsigma8='anal'
        sigmavv1_squre=np.abs(sigmavv_squre)
        sigmavv2_squre=sigmavv1_squre
        sigmavv3_squre=sigmavv1_squre     
    if(ps_type=='den-02 mom-0 crs-1'):
        fsigma8,  b1sigma8,b2sigma8,b3nlsigma8,   b1vsigma8,b2vsigma8,    sigmav1_squre,sigmav2_squre,sigmavv_squre = parm     
        # den ------------------- 
        bssigma8 ='anal'
        sigmav1_squre=np.abs(sigmav1_squre)
        sigmav2_squre=np.abs(sigmav2_squre)
        sigmav3_squre=sigmav1_squre
        # mom ------------------- 
        bsvsigma8 ='anal'
        b3nlvsigma8='anal'
        sigmavv1_squre=np.abs(sigmavv_squre)
        sigmavv2_squre=sigmavv1_squre
        sigmavv3_squre=sigmavv1_squre      
    return [fsigma8,  b1sigma8,b2sigma8,bssigma8,b3nlsigma8,  b1vsigma8,b2vsigma8,bsvsigma8,b3nlvsigma8,   sigmav1_squre,sigmav2_squre,sigmav3_squre,    sigmavv1_squre,sigmavv2_squre,sigmavv3_squre]
def CHI2(params,    ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type , Model_type):
    params = ParmPSmod_Fun(params, ps_type)
    Pk_modc, kmodc, Pk_mod , kmod =PkmodMulti_Fun(params,OnlyL0,Sig8_fid, k_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type )
    if(OnlyL0):
        VC=np.dot((Pk_obs[ind_fit]-Pk_modc[ind_fit]),Cov_inv)
        chi_squared=np.dot(VC,(Pk_obs[ind_fit]-Pk_modc[ind_fit]))
    if(not OnlyL0):
        Pk_obs,Pk_modc=PSmulti_pickup(ind_fit,ps_type,Pk_obs,Pk_modc)
        VC=np.dot((Pk_obs -Pk_modc ),Cov_inv)
        chi_squared=np.dot(VC,(Pk_obs -Pk_modc ))           
    return chi_squared
def tPDF(params,    Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type ,Model_type):
    chi_squared=CHI2(params,    ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type ,Model_type) 
    #print( chi_squared)
    Pb   = np.log(  Cp*(1.0+chi_squared/(Nmock-1.0))) * (-Nmock/2.0)         
    return -Pb
def Box_CoxFUN(params ,BC_nu,BC_d,Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs,Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type ,Model_type):
    params = ParmPSmod_Fun(params,ps_type)
    Pk_modc, kmodc,  Pk_mod , kmod  =PkmodMulti_Fun(params,OnlyL0,Sig8_fid, k_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type )
    if(OnlyL0):    
        XC=Pk_modc[ind_fit]+BC_d
        Pk_mod=np.zeros(len(XC))
        for i in range(len(XC)):
            if(XC[i]<=0):
                Pk_mod[i]=(-abs(XC[i]) **BC_nu[i]-1.0)/BC_nu[i]
            else:
                Pk_mod[i]=(XC[i] **BC_nu[i]-1.0)/BC_nu[i]
        #Pk_mod  =((Pk_mod[ind_fit]+BC_d)**BC_nu-1.0)/BC_nu
        VC   = np.dot((Pk_obs-Pk_mod),Cov_inv)
        chi2 = np.dot(VC,(Pk_obs-Pk_mod))
        Pb   = np.log(  Cp*(1.0+chi2/(Nmock-1.0))) * (-Nmock/2.0) 
    if(not OnlyL0): 
        tmp,Pk_modc=PSmulti_pickup(ind_fit,ps_type,Pk_modc,Pk_modc)
        XC=Pk_modc+BC_d
        Pk_mod=np.zeros(len(XC))
        for i in range(len(XC)):
            if(XC[i]<=0):
                Pk_mod[i]=(-abs(XC[i]) **BC_nu[i]-1.0)/BC_nu[i]
            else:
                Pk_mod[i]=(XC[i] **BC_nu[i]-1.0)/BC_nu[i]
        #Pk_mod  =((Pk_mod[ind_fit]+BC_d)**BC_nu-1.0)/BC_nu        
        VC   = np.dot((Pk_obs-Pk_mod),Cov_inv)
        chi2 = np.dot(VC,(Pk_obs-Pk_mod))
        Pb   = np.log(  Cp*(1.0+chi2/(Nmock-1.0))) * (-Nmock/2.0) 
    #print(  chi2 )        
    return -Pb
def ZeroPoint( S, Cinv,   x ):
    Nx=np.sqrt(    np.dot(np.dot(x,Cinv) ,x) + 1./Sig_ZP**2 )
    Ny=            np.dot(np.dot(S,Cinv) ,x)
    Vec=      -0.5*np.dot(np.dot(S,Cinv) ,S) + 0.5*Ny**2/Nx**2  
    ln_like=  - 0.5*np.log(Nx**2*Sig_ZP**2) + Vec
    return ln_like
def zpPDF(params    ,x,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type):
    params = ParmPSmod_Fun(params,ps_type)
    Pk_modc, kmodc, Pk_mod , kmod =PkmodMulti_Fun(params,OnlyL0,Sig8_fid, k_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type )
    Pk_obs,Pk_modc=PSmulti_pickup(ind_fit,ps_type,Pk_obs,Pk_modc)
    x,tmp=PSmulti_pickup(ind_fit,ps_type,x,x)
    S = Pk_obs -Pk_modc
    x[x==0]=0
    x[x==1]=1.
    x[x==2]=1/np.sqrt(Sig_ZP)
    Cov_inv=Cov_inv* (Nmock-2.-len(S))/(Nmock-1.)# hatlap effect. 
    like = ZeroPoint(S,Cov_inv,x)
    #print(-like)
    return -like 
# 5. the posterior and prior of MCMC:--------------------------------------
def lnprior(parm,ps_type):
    if(ps_type=='den-024 mom-024')or(ps_type=='den-024 mom-02')or(ps_type=='den-02 mom-02')or(ps_type=='den-024 mom-024 crs-13')or(ps_type=='den-02 mom-02 crs-1'):
        fsigma8,  b1sigma8,b2sigma8,b3nlsigma8,  b1vsigma8,   sigmav1_squre,sigmav2_squre,sigmavv1_squre,sigmavv2_squre = parm             
        if(0.  <fsigma8<1.5): pdf=1.
        else:return -np.inf
        if(0.  <b1sigma8<3.):pdf=1.
        else:return -np.inf
        if(-5.  <b2sigma8<5.):pdf=1.
        else:return -np.inf
        if(-5.  <b3nlsigma8<5.):pdf=1.
        else:return -np.inf
        if(0.  <b1vsigma8<3.):pdf=1.
        else:return -np.inf
        if( 0.<sigmav1_squre<350.):pdf=1.
        else:return -np.inf
        if( 0.<sigmav2_squre<350.):pdf=1.
        else:return -np.inf
        if( 0.<sigmavv1_squre<350.):pdf=1.
        else:return -np.inf
        return 0.0
        if( 0.<sigmavv2_squre<350.):pdf=1.
        else:return -np.inf
        return 0.0
    if(ps_type=='den-02 mom-0')or(ps_type=='den-024 mom-0')or(ps_type=='den-0 mom-0 crs-1'):
        fsigma8,  b1sigma8,b2sigma8,b3nlsigma8,  b1vsigma8,   sigmav1_squre,sigmav2_squre,sigmavv_squre = parm                 
        if(0.  <fsigma8<1.5): pdf=1.
        else:return -np.inf
        if(0.  <b1sigma8<3.):pdf=1.
        else:return -np.inf
        if(-5.  <b2sigma8<5.):pdf=1.
        else:return -np.inf
        if(-5.  <b3nlsigma8<5.):pdf=1.
        else:return -np.inf
        if(0.  <b1vsigma8<3.):pdf=1.
        else:return -np.inf
        if( 0.<sigmav1_squre<350.):pdf=1.
        else:return -np.inf
        if( 0.<sigmav2_squre<350.):pdf=1.
        else:return -np.inf
        if( 0.<sigmavv_squre<350.):pdf=1.
        else:return -np.inf
        return 0.0
    if(ps_type=='den-02 mom-0 crs-1'):
        fsigma8,  b1sigma8,b2sigma8,b3nlsigma8,   b1vsigma8,b2vsigma8,    sigmav1_squre,sigmav2_squre,sigmavv_squre = parm     
        if(0.  <fsigma8<1.5): pdf=1.
        else:return -np.inf
        if(0.  <b1sigma8<3.):pdf=1.
        else:return -np.inf
        if(-5.  <b2sigma8<5.):pdf=1.
        else:return -np.inf
        if(-5.  <b3nlsigma8<5.):pdf=1.
        else:return -np.inf
        if(0.  <b1vsigma8<3.):pdf=1.
        else:return -np.inf
        if(-3.  <b2vsigma8<3.):pdf=1.
        else:return -np.inf
        if( 0.<sigmav1_squre<350.):pdf=1.
        else:return -np.inf
        if( 0.<sigmav2_squre<350.):pdf=1.
        else:return -np.inf
        if( 0.<sigmavv_squre<350.):pdf=1.
        else:return -np.inf
        return 0.0
    if(ps_type=='den-0 mom-0'):
        fsigma8,  b1sigma8,b2sigma8,             b1vsigma8,   sigmav1_squre,sigmav2_squre,sigmavv_squre = parm     
        if(0.  <fsigma8<1.5): pdf=1.
        else:return -np.inf
        if(0.  <b1sigma8<3.):pdf=1.
        else:return -np.inf
        if(0.5  <b2sigma8<5.):pdf=1.
        else:return -np.inf
        if(0.  <b1vsigma8<3.):pdf=1.
        else:return -np.inf
        if( 0.<sigmav1_squre<550.):pdf=1.
        else:return -np.inf
        if( 0.<sigmav2_squre<550.):pdf=1.
        else:return -np.inf
        if( 0.<sigmavv_squre<550.):pdf=1.
        else:return -np.inf
        return 0.0

# 5.1: the post of MCMC:
def lnpost_tPDF(params    ,   Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type ,Model_type  ):
    prior = lnprior(params,ps_type  )
    if not np.isfinite(prior):
        return -np.inf
    like = tPDF(params ,Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type  )
    return prior - like
def lnpost_chi2(params    ,   Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type ,Model_type  ):
    prior = lnprior(params,ps_type  )
    if not np.isfinite(prior):
        return -np.inf
    like = -0.5* CHI2(params ,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type  )
    return prior + like
def lnpost_zp(params    , x,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type  ):
    prior = lnprior(params,ps_type )
    if not np.isfinite(prior):
        return -np.inf
    like = -1. *  zpPDF(params    ,x,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type )
    return prior + like
# 5.2: the post of MCMC for box-cox trans:
def lnpost_bc(params ,BC_nu,BC_d,Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs,Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type   ):
    prior = lnprior(params,ps_type  )
    if not np.isfinite(prior):
        return -np.inf
    like = Box_CoxFUN(params,BC_nu,BC_d,Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs,Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type   )
    return prior - like

# 5.3: fit func:---------------------------------------------------------------
def Param_Estimate(fit_minK,fit_maxK,Optp ,MCp,PS_type,fit_tech,strategy,OnlyL0,Sig8_fid,k_obs,Pk_obs,k_mocks,Pk_mocks,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,  az,Hz,Dz, WFCon_type,Model_type):
    print( '\n Important: make sure the mocks should have the same k-bins as the data! \n')
    print( '   Important: do not use the 1st k-bins for fitting when setting fit_minK! \n')
    print( 'PS_type  = ',PS_type ) 
    print( 'Only l=0 = ',OnlyL0)  
    print( 'Win_tech = ',WFCon_type  )
    print( 'K interval=',fit_minK,fit_maxK)
# 6.1 choosing fitting points:---------------------------
    ind_fit=np.where((k_obs>=fit_minK) & (k_obs<=fit_maxK))[0]    
    ind_mock_fit=np.where((k_mocks>=fit_minK) & (k_mocks<=fit_maxK))[0]
    Pk_mocks_fit=Pk_mocks[ind_mock_fit ,:]     
    if(not OnlyL0):
        Pmk=[]
        for i in range(len(Pk_mocks[0,:])):       
            Pkmk,Pkmk=PSmulti_pickup(ind_mock_fit,PS_type,Pk_mocks[:,i],Pk_mocks[:,i])
            Pmk.append(Pkmk)
        Pmks=np.zeros((len(Pmk[0]),len(Pmk)))    
        for i in range(len(Pmk)):
            Pmks[:,i]=Pmk[i]   
        Pk_mocks_fit=Pmks
    Nk=len(Pk_mocks_fit)
    Nmock = len(Pk_mocks_fit[0,:])   
# 6.2 chi2 minimization:---------------------------------
    if(fit_tech=='Chi2'):
        Cp=1.0
        print( 'fit_tech = ',fit_tech  )
        Cov  = np.cov(Pk_mocks_fit) #Cov=np.eye( Nk )  *  np.diag(np.cov(Pk_mocks_fit))#
        Cov_inv = np.linalg.inv(Cov) #np.eye( Nk )# 
        # 6.3.1 Optimize: 
        if(strategy=='Optimize'):
            print( 'strategy = Optimize\n')          
            outpf    = sp.optimize.minimize(tPDF,Optp,args=(Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type,Model_type ),method='Nelder-Mead',tol=10.**(-100))
            chisq    = CHI2(outpf.x,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type ,Model_type)
            if(Model_type=='lin'):outpf.x=np.abs(outpf.x)
            return    outpf.x,    chisq  
        if(strategy=='MCMC'):
            print( 'strategy = MCMC\n')
            if(Model_type=='nonlin'):
                if(PS_type=='den-024 mom-024')or(PS_type=='den-024 mom-02')or(PS_type=='den-02 mom-02')or(PS_type=='den-024 mom-024 crs-13')or(PS_type=='den-02 mom-02 crs-1'):
                    ndim, nwalkers = 9, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.05*(np.random.rand()-0.5)+Optp[4]),
                              (0.1*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]),
                              (0.1*(np.random.rand()-0.5)+Optp[7]),
                              (0.1*(np.random.rand()-0.5)+Optp[8])] for i in range(nwalkers)] 
                if(PS_type=='den-02 mom-0')or(PS_type=='den-024 mom-0')or(PS_type=='den-0 mom-0 crs-1'):
                    ndim, nwalkers = 8, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.05*(np.random.rand()-0.5)+Optp[4]),
                              (0.1*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]),
                              (0.1*(np.random.rand()-0.5)+Optp[7]) ] for i in range(nwalkers)] 
                if(PS_type=='den-02 mom-0 crs-1'):
                    ndim, nwalkers = 9, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.05*(np.random.rand()-0.5)+Optp[4]),
                              (0.05*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]),
                              (0.1*(np.random.rand()-0.5)+Optp[7]),
                              (0.1*(np.random.rand()-0.5)+Optp[8])] for i in range(nwalkers)] 
                if(PS_type=='den-0 mom-0'):
                    ndim, nwalkers = 7, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.1*(np.random.rand()-0.5)+Optp[4]),
                              (0.1*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]) ] for i in range(nwalkers)]     
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_tPDF, args=[Cp,Nmock, ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type,Model_type ])
            print('   ... mcmc initial sampling')
            pos, prob, state = sampler.run_mcmc(begin, MCp[0])
            sampler.reset()
            print('   ... mcmc final sampling')
            sampler.run_mcmc(pos, MCp[1])
            return sampler.flatchain
    if(fit_tech=='ZP'):
        print( 'fit_tech = ',fit_tech,'Sig_ZP=',Sig_ZP,'\n'  )
        Cov  = np.cov(Pk_mocks_fit)  
        Cov_inv = np.linalg.inv(Cov)  
        print( 'Note: The x and Sig_ZP should be updated for different data ! ! !\n'  )
        x=np.zeros(len(Pk_obs)) 
        x[len(x)//3:2*len(x)//3]=1
        x[2*len(x)//3: ]=2 
        # 6.3.1 Optimize: 
        if(strategy=='Optimize'):
            print( 'strategy = Optimize\n')          
            outpf    = sp.optimize.minimize(zpPDF,Optp,args=(x,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type,Model_type  ),method='Nelder-Mead',tol=10.**(-200))
            chisq    = CHI2(outpf.x,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type,Model_type )
            if(Model_type=='lin'):outpf.x=np.abs(outpf.x)
            return    outpf.x,    chisq  
        if(strategy=='MCMC'):
            print( 'strategy = MCMC\n')
            if(Model_type=='nonlin'):
                if(PS_type=='den-024 mom-024')or(PS_type=='den-024 mom-02')or(PS_type=='den-02 mom-02')or(PS_type=='den-024 mom-024 crs-13')or(PS_type=='den-02 mom-02 crs-1'):
                    ndim, nwalkers = 9, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.05*(np.random.rand()-0.5)+Optp[4]),
                              (0.1*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]),
                              (0.1*(np.random.rand()-0.5)+Optp[7]),
                              (0.1*(np.random.rand()-0.5)+Optp[8])] for i in range(nwalkers)] 
                if(PS_type=='den-02 mom-0')or(PS_type=='den-024 mom-0')or(PS_type=='den-0 mom-0 crs-1'):
                    ndim, nwalkers = 8, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.05*(np.random.rand()-0.5)+Optp[4]),
                              (0.1*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]),
                              (0.1*(np.random.rand()-0.5)+Optp[7]) ] for i in range(nwalkers)] 
                if(PS_type=='den-02 mom-0 crs-1'):
                    ndim, nwalkers = 9, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.05*(np.random.rand()-0.5)+Optp[4]),
                              (0.05*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]),
                              (0.1*(np.random.rand()-0.5)+Optp[7]),
                              (0.1*(np.random.rand()-0.5)+Optp[8])] for i in range(nwalkers)] 
                if(PS_type=='den-0 mom-0'):
                    ndim, nwalkers = 7, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.1*(np.random.rand()-0.5)+Optp[4]),
                              (0.1*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]) ] for i in range(nwalkers)]   
            if(Model_type=='lin'):
                ndim, nwalkers = 4, 30
                begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                          (0.05*(np.random.rand()-0.5)+Optp[1]),                       
                          (0.1*(np.random.rand()-0.5)+Optp[2]),
                          (0.1*(np.random.rand()-0.5)+Optp[3]) ] for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_zp, args=[x,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs, Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type,Model_type ])
            print('   ... mcmc initial sampling')
            pos, prob, state = sampler.run_mcmc(begin, MCp[0])
            sampler.reset()
            print('   ... mcmc final sampling')
            sampler.run_mcmc(pos, MCp[1])
            return sampler.flatchain        
    if(fit_tech=='BC'):
        print( 'fit_tech = ',fit_tech  )       
        Cov0  = np.cov(Pk_mocks_fit)  
        Cov_inv0 = np.linalg.inv(Cov0)
        Pk_obs0= Pk_obs+1.-1.        
        if( OnlyL0):Pk_obs = Pk_obs[ind_fit]    
        if(not OnlyL0):Pk_obs,Pk_obs=PSmulti_pickup(ind_fit,PS_type,Pk_obs,Pk_obs)      
        lmbdas = np.linspace(-0.8, 1.3,200)
        llf    = np.zeros(lmbdas.shape, dtype= float)
        BC_nu  = np.zeros(Nk)
        BC_d   = np.zeros(Nk)
        for ik in range(Nk):
            if(min(Pk_mocks_fit[ik,:]) <=0.0):
                BC_d[ik]=abs( min(Pk_mocks_fit[ik,:]) )+0.01*(max(Pk_mocks_fit[ik,:])-min(Pk_mocks_fit[ik,:]))
            if(min(Pk_mocks_fit[ik,:]) >0.0):
                BC_d[ik]=0.0
            #BC_d[ik]=0.001-np.min(Pk_mocks_fit.flatten())
            for ii, lmbda in enumerate(lmbdas):
                llf[ii] = stats.boxcox_llf(lmbda, Pk_mocks_fit[ik,:]+BC_d[ik])
            BC_nu[ik]=lmbdas[llf==max(llf)]
            if(  (Pk_obs[ik]+BC_d[ik])>0.0  ):
                Pk_obs[ik]     = (( Pk_obs[ik]         + BC_d[ik] )**BC_nu[ik]-1.0)/BC_nu[ik]
            else:
                print( '======-----------   ********   -------------=====')
                print( 'Warning: Pk+delta<0, the results may be biased!!!')
                print( 'please increasing the BC_d by increasing 0.01 to a larger value!!')
                print( '======-----------   ********   -------------=====')
                Pk_obs[ik]     = (-abs( Pk_obs[ik]     + BC_d[ik] )**BC_nu[ik]-1.0)/BC_nu[ik]
            Pk_mocks_fit[ik,:] = (( Pk_mocks_fit[ik,:] + BC_d[ik] )**BC_nu[ik]-1.0)/BC_nu[ik]
        # inverse covaraince matrix:        
        Cov  = np.cov(Pk_mocks_fit)  
        Cov_inv = np.linalg.inv(Cov) 
        #Cp=det**(-0.5)*((Nmock-1)*math.pi)**(-Nk/2.0)*(sp.special.gamma( Nmock/2.0 ))/(sp.special.gamma( (Nmock-Nk)/2.0 ))
        Cp=1.0        
        if(strategy=='Optimize'):
            print( 'strategy = Optimize\n')        
            outpf    = sp.optimize.minimize(Box_CoxFUN, Optp,args=( BC_nu,BC_d,Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs,Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type ,Model_type ),method='Nelder-Mead',tol=10.**(-100))
            chisq    = CHI2(outpf.x,ind_fit,Cov_inv0,OnlyL0,Sig8_fid, k_obs, Pk_obs0,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type,Model_type)
            if(Model_type=='lin'):outpf.x=np.abs(outpf.x)
            return    outpf.x,    chisq         
        if(strategy=='MCMC'):
            print( 'strategy = MCMC\n')
            if(Model_type=='nonlin'):
                if(PS_type=='den-024 mom-024')or(PS_type=='den-024 mom-02')or(PS_type=='den-02 mom-02')or(PS_type=='den-024 mom-024 crs-13')or(PS_type=='den-02 mom-02 crs-1'):
                    ndim, nwalkers = 9, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.05*(np.random.rand()-0.5)+Optp[4]),
                              (0.1*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]),
                              (0.1*(np.random.rand()-0.5)+Optp[7]),
                              (0.1*(np.random.rand()-0.5)+Optp[8])] for i in range(nwalkers)] 
                if(PS_type=='den-02 mom-0')or(PS_type=='den-024 mom-0')or(PS_type=='den-0 mom-0 crs-1'):
                    ndim, nwalkers = 8, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.05*(np.random.rand()-0.5)+Optp[4]),
                              (0.1*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]),
                              (0.1*(np.random.rand()-0.5)+Optp[7]) ] for i in range(nwalkers)] 
                if(PS_type=='den-02 mom-0 crs-1'):
                    ndim, nwalkers = 9, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.05*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.05*(np.random.rand()-0.5)+Optp[4]),
                              (0.05*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]),
                              (0.1*(np.random.rand()-0.5)+Optp[7]),
                              (0.1*(np.random.rand()-0.5)+Optp[8])] for i in range(nwalkers)] 
                if(PS_type=='den-0 mom-0'):
                    ndim, nwalkers = 7, 30
                    begin = [[(0.05*(np.random.rand()-0.5)+Optp[0]),
                              (0.05*(np.random.rand()-0.5)+Optp[1]),
                              (0.01*(np.random.rand()-0.5)+Optp[2]),
                              (0.05*(np.random.rand()-0.5)+Optp[3]),
                              (0.1*(np.random.rand()-0.5)+Optp[4]),
                              (0.1*(np.random.rand()-0.5)+Optp[5]),
                              (0.1*(np.random.rand()-0.5)+Optp[6]) ] for i in range(nwalkers)]   
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_bc, args=[BC_nu,BC_d,Cp,Nmock,ind_fit,Cov_inv,OnlyL0,Sig8_fid, k_obs,Pk_obs, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type,Model_type ])
            print('   ... mcmc initial sampling')
            pos, prob, state = sampler.run_mcmc(begin, MCp[0])
            sampler.reset()
            print('   ... mcmc final sampling')
            sampler.run_mcmc(pos, MCp[1])
            return sampler.flatchain   
################          The end of Sec.2          ###########################
###############################################################################        
        
        
        
        
        
        
        
        












###############################################################################
######                                                                   ######
######                            Sec 3. Plots                           ######
######                                                                   ######
###############################################################################                 
def MCplot(mean,chain,ps_type,savr):
    import pandas as pd
    from chainconsumer import ChainConsumer,Chain,ChainConfig,PlotConfig,Truth
    
    if(ps_type=='den-024 mom-024')or(ps_type=='den-024 mom-02')or(ps_type=='den-02 mom-02')or(ps_type=='den-024 mom-024 crs-13')or(ps_type=='den-02 mom-02 crs-1'):
        df = pd.DataFrame({ '$f\sigma_8$':chain[:,0],   
                            '$b^{\delta}_1\sigma_8$': chain[:,1], 
                            '$b^{\delta}_2\sigma_8$': chain[:,2],
                            '$b^{\delta}_{3nl}\sigma_8$': chain[:,3],
                            '$b^{p}_1\sigma_8$': chain[:,4],                            
                            '$\sigma^2_{\delta,vT}$': chain[:,5],
                            '$\sigma^2_{\delta,vS}$': chain[:,6],
                            '$\sigma^2_{p,vT}$': chain[:,7],
                            '$\sigma^2_{p,vS}$': chain[:,8] })  
    if(ps_type=='den-02 mom-0')or(ps_type=='den-024 mom-0')or(ps_type=='den-0 mom-0 crs-1'):
        df = pd.DataFrame({ '$f\sigma_8$':chain[:,0],   
                            '$b^{\delta}_1\sigma_8$': chain[:,1], 
                            '$b^{\delta}_2\sigma_8$': chain[:,2],
                            '$b^{\delta}_{3nl}\sigma_8$': chain[:,3],
                            '$b^{p}_1\sigma_8$': chain[:,4],                        
                            '$\sigma^2_{\delta,vT}$': chain[:,5],
                            '$\sigma^2_{\delta,vS}$': chain[:,6],
                            '$\sigma^2_{p,v}$': chain[:,7] })       
    if(ps_type=='den-0 mom-0'):
        df = pd.DataFrame({ '$f\sigma_8$':chain[:,0],   
                            '$b^{\delta}_1\sigma_8$': chain[:,1], 
                            '$b^{\delta}_2\sigma_8$': chain[:,2],                
                            '$b^{p}_1\sigma_8$': chain[:,3],                            
                            '$\sigma^2_{\delta,vT}$': chain[:,4],
                            '$\sigma^2_{\delta,vS}$': chain[:,5],
                            '$\sigma^2_{p,v}$': chain[:,6] })      
    if(ps_type=='den-02 mom-0 crs-1'):
        df = pd.DataFrame({ '$f\sigma_8$':chain[:,0],   
                            '$b^{\delta}_1\sigma_8$': chain[:,1], 
                            '$b^{\delta}_2\sigma_8$': chain[:,2],
                            '$b^{\delta}_{3nl}\sigma_8$': chain[:,3],
                            '$b^{p}_1\sigma_8$': chain[:,4], 
                            '$b^{p}_2\sigma_8$': chain[:,5],
                            '$\sigma^2_{\delta,vT}$': chain[:,6],
                            '$\sigma^2_{\delta,vS}$': chain[:,7],
                            '$\sigma^2_{p,v}$': chain[:,8] })
    c = ChainConsumer()
    c.add_chain(Chain(samples=df , name="sc",color="#00bfa5",plot_cloud=True ))
    c.set_override(ChainConfig(sigmas=[0,1,1.5,2,2.5]))
    c.add_truth(Truth(location={'$f\sigma_8$': mean } ))
    c.set_plot_config( PlotConfig( flip=False,summary_font_size=14, tick_font_size=14,label_font_size=19 ))
    fig = c.plotter.plot(figsize=(16,16),filename=savr ) 
    plt.show() 
 
def Plot_l0(params , Nk,fit_minK,fit_maxK,Sig8_fid, k_data,Pk_data,ekPk,ePk, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type,iplt ): 
    params = ParmPSmod_Fun(params,ps_type)
    if('den' in ps_type ) and('mom' not in ps_type):
        Pkmodc, kmodc, Pkmod , kmod =PkmodMulti_Fun(params,True,Sig8_fid, k_data, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type )
        indkobs=np.where((k_data[:Nk]>=fit_minK)&(k_data[:Nk]<=fit_maxK))[0] 
        indkmodc=np.where((kmodc>=fit_minK)&(kmodc<=fit_maxK))[0]
        indkmod=np.where((kmod>=fit_minK)&(kmod<=fit_maxK))[0]
        plt.figure(1+iplt,figsize=(6.5,4.5))
        fsiz=25;tsiz=18
        plt.errorbar(k_data[indkobs],(k_data *Pk_data)[indkobs],ekPk[indkobs],ls='',marker='s' ,ms=6,label='Measurements')
        plt.plot(kmodc[indkmodc],(kmodc*Pkmodc)[indkmodc],label='Model' ); 
        plt.plot(kmod[indkmod],(kmod*Pkmod)[indkmod],label='unconv-Model' ); 
        plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
        plt.ylabel('$kP^{\delta}_{0}(k)$', fontsize=fsiz)
        plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
        plt.legend( loc='upper right', fontsize=tsiz)        
    if('mom' in ps_type ) and('den' not in ps_type):
        Pkmodc, kmodc, Pkmod , kmod =PkmodMulti_Fun(params,True,Sig8_fid, k_data, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type )
        indkobs=np.where((k_data[:Nk]>=fit_minK)&(k_data[:Nk]<=fit_maxK))[0] 
        indkmodc=np.where((kmodc>=fit_minK)&(kmodc<=fit_maxK))[0]
        indkmod=np.where((kmod>=fit_minK)&(kmod<=fit_maxK))[0]
        plt.figure(2+iplt,figsize=(6.5,4.5))
        fsiz=25;tsiz=18
        plt.yscale('log')
        plt.errorbar(k_data[indkobs],(Pk_data)[indkobs],ePk[indkobs],ls='',marker='s' ,ms=6,label='Measurements')
        plt.plot(kmodc[indkmodc],( Pkmodc)[indkmodc],label='Model' ); 
        plt.plot(kmod[indkmod],( Pkmod)[indkmod],label='unconv-Model' ); 
        plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
        plt.ylabel('$P^{p}_{0}(k)$', fontsize=fsiz)
        plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
        plt.legend( loc='upper right', fontsize=tsiz)    
    if('mom' in ps_type ) and('den'   in ps_type):
        Pkmodc, kmodc, Pkmod , kmod =PkmodMulti_Fun(params,True,Sig8_fid, k_data, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,ps_type, az,Hz,Dz, WFCon_type,Model_type )
        Pkp_data=Pk_data[Nk:]
        Pkpmodc =Pkmodc[Nk:]
        Pkpmod  =Pkmod[len(kmod)//2:]
        ePk     =ePk[Nk:]
        Pk_data=Pk_data[:Nk]
        Pkmodc =Pkmodc[:Nk]
        Pkmod  =Pkmod[:len(kmod)//2]      
        ekPk    =ekPk[:Nk]
        k_data =k_data[:Nk]
        kmodc=kmodc[:Nk]
        kmod =kmod[:len(kmod)//2]
        indkobs=np.where((k_data[:Nk]>=fit_minK)&(k_data[:Nk]<=fit_maxK))[0] 
        indkmodc=np.where((kmodc>=fit_minK)&(kmodc<=fit_maxK))[0]
        indkmod=np.where((kmod>=fit_minK)&(kmod<=fit_maxK))[0]
        plt.figure(1+iplt,figsize=(6.5,4.5))
        fsiz=25;tsiz=18
        plt.errorbar(k_data[indkobs],(k_data *Pk_data)[indkobs],ekPk[indkobs],ls='',marker='s' ,ms=6,label='Measurements')
        plt.plot(kmodc[indkmodc],(kmodc*Pkmodc)[indkmodc],label='Model' ); 
        plt.plot(kmod[indkmod],(kmod*Pkmod)[indkmod],label='unconv-Model' ); 
        plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
        plt.ylabel('$kP^{\delta}_{0}(k)$  [ $h^{-2} Mpc^2$ ]', fontsize=fsiz)
        plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
        plt.legend( loc='upper right', fontsize=tsiz) 
        plt.figure(2+iplt,figsize=(6.5,4.5))
        fsiz=25;tsiz=18
        plt.yscale('log')
        plt.errorbar(k_data[indkobs],(Pkp_data)[indkobs],ePk[indkobs],ls='',marker='s' ,ms=6,label='Measurements')
        plt.plot(kmodc[indkmodc],( Pkpmodc)[indkmodc],label='Model' ); 
        plt.plot(kmod[indkmod],( Pkpmod)[indkmod],label='unconv-Model' ); 
        plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
        plt.ylabel('$P^{p}_{0}(k)$', fontsize=fsiz)
        plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
        plt.legend( loc='upper right', fontsize=tsiz)    
    
def Plot_denMulti(param_test ,Nk,fit_minK,fit_maxK,k_data,Pk_data,ePk,ekPk,Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,Model_type,iplt ):    
    param_test = ParmPSmod_Fun(param_test,PS_type)
    Ps_type = PS_type.split(' ')
    ps_type=[]
    mt=[]
    Nmt=[]
    for i in range(len(Ps_type) ):
        ps_type0=Ps_type[i].split('-')
        mt.append( ps_type0[1])
        ps_type.append(ps_type0[0]) 
        Nmt.append( len( ps_type0[1]))   
    Pkmodc , kmodc , Pkmod , kmod  =PkmodMulti_Fun(param_test,False,Sig8_fid, k_data, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,'den-024', az,Hz,Dz, WFCon_type,Model_type )
    Nkm=len(kmod)  //3
    k_data = k_data[:Nk]
    kmod   = kmod[:Nkm]
    kmodc  = kmodc[:Nk]
    indkmod= np.where((kmod   >=fit_minK)&(kmod   <=fit_maxK))[0]
    indkmodc= np.where((kmodc   >=fit_minK)&(kmodc   <=fit_maxK))[0]
    indkobs= np.where((k_data >=fit_minK)&(k_data <=fit_maxK))[0]   
    if('den' in ps_type) :     
        intd=np.where(np.array(ps_type)=='den')[0]
        mut=np.array(mt)[intd][0]      
        for i in range(len(mut)):
            if('0' in mut):
                mut=mut.replace('0','')
                plt.figure(1+iplt,figsize=(6.5,4.5))
                fsiz=25;tsiz=18
                plt.errorbar(k_data[indkobs],(k_data*Pk_data[0*Nk:1*Nk])[indkobs] ,ekPk[0*Nk:1*Nk][indkobs],ls='',marker='s' ,ms=6,label='Measurements')
                plt.plot(    kmodc[indkmodc],(kmodc  *Pkmodc[0*Nk:1*Nk])[indkmodc],label='Model' );
                #plt.plot(    kmod[indkmod]  ,(kmod   *Pkmod[0*Nkm:1*Nkm] )[indkmod],label='unconv-Model' );
                plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
                plt.ylabel('$kP^{\delta}_{0}(k)$  [ $h^{-2} Mpc^2$ ]', fontsize=fsiz)
                plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
                plt.legend( loc='upper right', fontsize=tsiz)
            if('2' in mut):
                mut=mut.replace('2','')
                plt.figure(2+iplt,figsize=(6.5,4.5))
                plt.errorbar(k_data[indkobs],(k_data*Pk_data[1*Nk:2*Nk])[indkobs] ,ekPk[1*Nk:2*Nk][indkobs],ls='',marker='s' ,ms=6,label='Measurements')
                plt.plot(    kmodc[indkmodc],(kmodc  *Pkmodc[1*Nk:2*Nk])[indkmodc],label='Model' );
                #plt.plot(    kmod[indkmod]  ,(kmod   *Pkmod[1*Nkm:2*Nkm] )[indkmod],label='unconv-Model' );
                plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
                plt.ylabel('$kP^{\delta}_{2}(k)$  [ $h^{-2} Mpc^2$ ]', fontsize=fsiz)
                plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
                plt.legend( loc='upper right', fontsize=tsiz)                   
            if('4' in mut):
                mut=mut.replace('4','')
                plt.figure(3+iplt,figsize=(6.5,4.5))  
                plt.errorbar(k_data[indkobs],(k_data*Pk_data[2*Nk:3*Nk])[indkobs] ,ekPk[2*Nk:3*Nk][indkobs],ls='',marker='s' ,ms=6,label='Measurements')
                plt.plot(    kmodc[indkmodc],(kmodc  *Pkmodc[2*Nk:3*Nk])[indkmodc],label='Model' );
                #plt.plot(    kmod[indkmod]  ,(kmod   *Pkmod[2*Nkm:3*Nkm] )[indkmod],label='unconv-Model' );
                plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
                plt.ylabel('$kP^{\delta}_{4}(k)$  [ $h^{-2} Mpc^2$ ]', fontsize=fsiz)
                plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
                plt.legend( loc='upper right', fontsize=tsiz)

def Plot_momMulti(param_test ,Nk,fit_minK,fit_maxK,k_data,Pk_data,ePk,ekPk,Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,Model_type,iplt ):    
    param_test = ParmPSmod_Fun(param_test,PS_type)
    Ps_type = PS_type.split(' ')
    ps_type=[]
    mt=[]
    Nmt=[]
    for i in range(len(Ps_type) ):
        ps_type0=Ps_type[i].split('-')
        mt.append( ps_type0[1])
        ps_type.append(ps_type0[0]) 
        Nmt.append( len( ps_type0[1]))   
    Pkmodc , kmodc , Pkmod , kmod  =PkmodMulti_Fun(param_test,False,Sig8_fid, k_data, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,'mom-024', az,Hz,Dz, WFCon_type,Model_type )
    Nkm=len(kmod)  //3
    k_data = k_data[:Nk]
    kmod   = kmod[:Nkm]
    kmodc  = kmodc[:Nk]
    indkmod= np.where((kmod   >=fit_minK)&(kmod   <=fit_maxK))[0]
    indkmodc= np.where((kmodc   >=fit_minK)&(kmodc   <=fit_maxK))[0]
    indkobs= np.where((k_data >=fit_minK)&(k_data <=fit_maxK))[0]   
    if('mom' in ps_type) :     
        intd=np.where(np.array(ps_type)=='mom')[0]
        mut=np.array(mt)[intd][0]      
        for i in range(len(mut)):
            if('0' in mut):
                mut=mut.replace('0','')
                plt.figure(1+iplt,figsize=(6.5,4.5))
                fsiz=25;tsiz=18
                plt.yscale('log')
                plt.errorbar(k_data[indkobs],( Pk_data[0*Nk:1*Nk])[indkobs] ,ePk[0*Nk:1*Nk][indkobs],ls='',marker='s' ,ms=6,label='Measurements')
                plt.plot(    kmodc[indkmodc],( Pkmodc[0*Nk:1*Nk])[indkmodc],label='Model' );
                #plt.plot(    kmod[indkmod]  ,( Pkmod[0*Nkm:1*Nkm] )[indkmod],label='unconv-Model' );
                plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
                plt.ylabel('$P^{p}_{0}(k)$  [ $h^{-2} Mpc^2$ ]', fontsize=fsiz)
                plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
                plt.legend( loc='upper right', fontsize=tsiz)
            if('2' in mut):
                mut=mut.replace('2','')
                plt.figure(2+iplt,figsize=(6.5,4.5))
                plt.yscale('log')
                plt.errorbar(k_data[indkobs],( Pk_data[1*Nk:2*Nk])[indkobs] ,ePk[1*Nk:2*Nk][indkobs],ls='',marker='s' ,ms=6,label='Measurements')
                plt.plot(    kmodc[indkmodc],( Pkmodc[1*Nk:2*Nk])[indkmodc],label='Model' );
                #plt.plot(    kmod[indkmod]  ,( Pkmod[1*Nkm:2*Nkm] )[indkmod],label='unconv-Model' );
                plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
                plt.ylabel('$P^{p}_{2}(k)$  [ $h^{-2} Mpc^2$ ]', fontsize=fsiz)
                plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
                plt.legend( loc='upper right', fontsize=tsiz)                   
            if('4' in mut):
                mut=mut.replace('4','')
                plt.figure(3+iplt,figsize=(6.5,4.5)) 
                plt.yscale('log')
                plt.errorbar(k_data[indkobs],( Pk_data[2*Nk:3*Nk])[indkobs] ,ePk[2*Nk:3*Nk][indkobs],ls='',marker='s' ,ms=6,label='Measurements')
                plt.plot(    kmodc[indkmodc],( Pkmodc[2*Nk:3*Nk])[indkmodc],label='Model' );
                #plt.plot(    kmod[indkmod]  ,( Pkmod[2*Nkm:3*Nkm] )[indkmod],label='unconv-Model' );
                plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
                plt.ylabel('$P^{p}_{4}(k)$  [ $h^{-2} Mpc^2$ ]', fontsize=fsiz)
                plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
                plt.legend( loc='upper right', fontsize=tsiz)                

def Plot_crsMulti(param_test ,Nk,fit_minK,fit_maxK,k_data,Pk_data,ePk,ekPk,Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,Model_type,iplt ):    
     param_test = ParmPSmod_Fun(param_test,PS_type)
     Ps_type = PS_type.split(' ')
     ps_type=[]
     mt=[]
     Nmt=[]
     for i in range(len(Ps_type) ):
         ps_type0=Ps_type[i].split('-')
         mt.append( ps_type0[1])
         ps_type.append(ps_type0[0]) 
         Nmt.append( len( ps_type0[1]))   
     Pkmodc , kmodc , Pkmod , kmod  =PkmodMulti_Fun(param_test,False,Sig8_fid, k_data, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,'crs-13', az,Hz,Dz, WFCon_type,Model_type )
     Nkm=len(kmod)  //2 
     k_data = k_data[:Nk]
     kmod   = kmod[:Nkm]
     kmodc  = kmodc[:Nk]
     indkmod= np.where((kmod   >=fit_minK)&(kmod   <=fit_maxK))[0]
     indkmodc= np.where((kmodc   >=fit_minK)&(kmodc   <=fit_maxK))[0]
     indkobs= np.where((k_data >=fit_minK)&(k_data <=fit_maxK))[0]   
     if('crs' in ps_type) :     
         intd=np.where(np.array(ps_type)=='crs')[0]
         mut=np.array(mt)[intd][0]      
         for i in range(len(mut)):
             if('1' in mut):
                 mut=mut.replace('1','')
                 plt.figure(1+iplt,figsize=(6.5,4.5))
                 fsiz=25;tsiz=18
                 plt.errorbar(k_data[indkobs],(k_data*Pk_data[0*Nk:1*Nk])[indkobs] ,ekPk[0*Nk:1*Nk][indkobs],ls='',marker='s' ,ms=6,label='Measurements')
                 plt.plot(    kmodc[indkmodc],(kmodc  *Pkmodc[0*Nk:1*Nk])[indkmodc],label='Model' );
                 #plt.plot(    kmod[indkmod]  ,(kmod   *Pkmod[0*Nkm:1*Nkm] )[indkmod],label='unconv-Model' );
                 plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
                 plt.ylabel('$kP^{\delta p}_{1}(k)$  [ $h^{-2} Mpc^2$ ]', fontsize=fsiz)
                 plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
                 plt.legend( loc='upper right', fontsize=tsiz)
             if('3' in mut):
                 mut=mut.replace('3','')
                 plt.figure(2+iplt,figsize=(6.5,4.5))
                 plt.errorbar(k_data[indkobs],(k_data*Pk_data[1*Nk:2*Nk])[indkobs] ,ekPk[1*Nk:2*Nk][indkobs],ls='',marker='s' ,ms=6,label='Measurements')
                 plt.plot(    kmodc[indkmodc],(kmodc  *Pkmodc[1*Nk:2*Nk])[indkmodc],label='Model' );
                 #plt.plot(    kmod[indkmod]  ,(kmod   *Pkmod[1*Nkm:2*Nkm] )[indkmod],label='unconv-Model' );
                 plt.xlabel('k  [ $h Mpc^{-1}$ ]', fontsize=fsiz)
                 plt.ylabel('$kP^{\delta p}_{3}(k)$  [ $h^{-2} Mpc^2$ ]', fontsize=fsiz)
                 plt.xticks(fontsize=tsiz);plt.yticks(fontsize=tsiz)
                 plt.legend( loc='upper right', fontsize=tsiz)                   
                              
################          The end of Sec.3          ###########################
###############################################################################        
           
        
