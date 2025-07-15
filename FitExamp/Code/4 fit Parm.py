import numpy as np
from PSmodFun import *
from PSfitFun import * 
#==========================================================================
#                           INITIAL SETTING
#--------------------------------------------------------------------------
# 1. INITIAL SETTING:
OmegaM    = 0.3 
OmegaA    = 1.- OmegaM
hb        = 0.68
Hub       = 100.
ombh2     = 0.02210272
omch2     = 0.11661728
ns        = 0.96 
As        = 2.2154397054279834e-09
KmodLim   = 1.
NKmodbin  = 437
RsFeff    = 0.1
Gamma_fid = 0.55
az        = 1./(1.+RsFeff)
Hz        = Hub*Ez(RsFeff, OmegaM, OmegaA, 0., -1., 0., 0.)
Dz        = Dz_Fun(RsFeff, OmegaM, OmegaA )
 

Model_type =  'nonlin' #   'lin'#  
 
PS_type    = 'den-02 mom-0 crs-1'
PS_measure =  'mockAve'#'survey'#     'mockML'    #                 
WFCon_type = 'Blake'   #'Beutler'#        # 'Ross'   #          is only for l=0, if using multiple, you have to use 'Black' .
OnlyL0     =  False     #  True #               # only fit l=0
strategy   = 'Optimize' #   'MCMC'  #     
fit_tech   =    'BC' #   'ZP' #   'Chi2'     #               
IntegCal   =  False   # True # 
fit_minK   =  0.025       
fit_maxK   =  0.3  
Nmock      =  599
MCparm     =  [500,4500]
file_dir   = '/Users/fei/WSP/Scie/CosmPS/FitExamp/Data/ProdPy/'



if(PS_type=='den-024 mom-024'):
    fit_minK =  0.065       
    fit_maxK =  0.2 
    Optparm  = [ 0.43954 , 1.1049 ,  1.3140 , -0.31700 , 0.969762, 71.8908 , 148.1969 , 16.07854911 , 88.91464184]
if(PS_type=='den-024 mom-02'):
    fit_minK =  0.03
    fit_maxK =  0.2 
    Optparm  = [ 0.46195 , 1.0819 , 1.42494 , -0.20868 , 0.976822, 62.0240 , 144.0757 , 24.43534985 , 89.16845613]
if(PS_type=='den-02 mom-02' ):
    fit_minK =  0.03
    fit_maxK =  0.25
    Optparm  = [ 0.46484 , 1.1077 , 1.27895 , -0.29375 , 0.971195, 60.3490 , 156.0479 , 27.53694611 , 82.57022397]
if(PS_type=='den-024 mom-0'):
    fit_minK =  0.025
    fit_maxK =  0.2 
    Optparm  = [ 0.45129 , 1.0915 , 1.42120 , -0.22419 , 0.683398, 67.3137 , 151.3205 , 2.152173e-12]
if(PS_type=='den-02 mom-0' ):
    fit_minK =  0.025
    fit_maxK =  0.3
    Optparm  = [ 0.47271 , 1.1241 , 1.19780 , -0.33496 , 0.670880, 62.8197 , 158.1764 , 2.072081e-12]
if(PS_type=='den-0 mom-0' ):
    fit_minK =  0.025   
    fit_maxK =  0.3 
    Optparm  = [ 0.45776 , 1.0891 , 1.37388 , 0.690666 , 75.95678, 339.667 , 9.0689902e-13]
if(PS_type=='den-0 mom-0 crs-1' ):
    fit_minK =  0.025   
    fit_maxK =  0.3 
    Optparm  = [ 0.46698 , 1.0612 , -0.1736 , 0.084692 , 0.809014, 66.9961 , 137.6588 , 37.2627931]
if(PS_type=='den-02 mom-0 crs-1' ):
    fit_minK =  0.025   
    fit_maxK =  0.3 
    Optparm  = [ 0.45924 , 1.0559 , -0.3860 , 0.183126 , 0.803765, -0.1614 , 76.28646 , 120.664242 , 36.80603667]
if(PS_type=='den-02 mom-02 crs-1' ):
    fit_minK =  0.03   
    fit_maxK =  0.25
    Optparm  = [ 0.43870 , 1.1226 , 1.16495 , -0.43981 , 0.936926, 63.4358 , 140.1625 , 21.4179908 , 69.13133373]
if(PS_type=='den-024 mom-024 crs-13' ):
    fit_minK =  0.065       
    fit_maxK =  0.2 
    Optparm  = [ 0.44464 , 1.1149 , 1.05261 , -0.43773 , 0.873770, 77.0300 , 129.1595 , 18.1654795 , 70.59401097]

 
 
#==============================================================================
#                                   MAIN CODE
# 1. read data:---------------------------------------------------------------- 
if(WFCon_type=='Ross' ): OnlyL0=True ;  
Nk,k_data,Pk_data,k_mock,Pk_mock,conv, kmodc,wf_rand=LoadPS_Fun(file_dir,Nmock,PS_type,WFCon_type,OnlyL0,PS_measure)  
Pave,   ePk,  ekPk =  PSmock_prop(k_mock,Pk_mock,fit_minK ,fit_maxK,PS_measure )
if(PS_measure =='mockAve')or(PS_measure =='mockML'): Pk_data= Pave
print(PS_measure)


# 2. Calculate the integrations:-----------------------------------------------
if(Model_type =='nonlin' ):
    if(IntegCal):PT,Sig8_fid,outdir=PSloop_Fun( KmodLim,NKmodbin,Hub*hb,ombh2,omch2,ns,As,file_dir)  
    if(not IntegCal):
        sst=np.load(file_dir+'INTEG_PL.npy',allow_pickle=True)
        PT,Sig8_fid=sst
if(Model_type =='lin' ):
    kcambNL,PcambNL,Sig8_fid=CAMB_Fun(1e-4,1.0,NKmodbin,RsFeff,Hub*hb,ombh2,omch2,ns, As, "mead2020")
    PT=np.zeros((2,len(kcambNL)));PT[0,:] =  kcambNL;PT[1,:] =  PcambNL 
    fit_maxK = 0.1 
print('Sig8_fid =',Sig8_fid,'\n') 
fsig8_fid=Fsigma8_Fun(Gamma_fid,RsFeff,Sig8_fid, OmegaM, OmegaA)
print('fsig8_fid=',fsig8_fid) 




# 3. Fit parms:----------------------------------------------------------------
k_obs        = k_data
Pk_obs       = Pk_data
k_mocks      = k_mock
Pk_mocks     = Pk_mock 
kmod_conv    = kmodc
WF_Kobs_Kmod = conv
WF_rand      = wf_rand
  
if(strategy =='Optimize'):
    fitparm, chi2s= Param_Estimate( fit_minK,fit_maxK,Optparm,MCparm,PS_type,fit_tech,strategy,OnlyL0,Sig8_fid,k_obs,Pk_obs,k_mocks,Pk_mocks,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,  az,Hz,Dz, WFCon_type,Model_type )
    if(Model_type=='nonlin'):
        print( 'fsigma8   =',  fitparm[0])
        print( fitparm[1:]  ) 
        print( 'CHI2      =',  chi2s   )  
    if(Model_type=='lin'):  
        fsigma8,bsigma8,sigvd,sigvv  =fitparm
        print( 'fsigma8   =', fsigma8)
        print( 'bsigma8 =', bsigma8)
        print( 'sigmav (not squre)  =', sigvd,sigvv  ) 
        print( 'CHI2      =', chi2s   )
    param_test = fitparm   
    if(OnlyL0)    : Plot_l0(param_test , Nk,fit_minK,fit_maxK,Sig8_fid, k_obs, Pk_obs,ekPk,ePk, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, az,Hz,Dz, WFCon_type,Model_type,0 )  
    if(not OnlyL0): 
        if('den' in PS_type)and('mom' not in PS_type)and('crs' not in PS_type):
            Plot_denMulti(param_test ,Nk,fit_minK,fit_maxK,k_data,Pk_data,ePk,ekPk,Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,Model_type,5 )    
        if('mom' in PS_type)and('den' not in PS_type)and('crs' not in PS_type):
            Plot_momMulti(param_test ,Nk,fit_minK,fit_maxK,k_data,Pk_data,ePk,ekPk,Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,Model_type,5 )    
        if('crs' in PS_type)and('den' not in PS_type)and('mom' not in PS_type):
            Plot_crsMulti(param_test ,Nk,fit_minK,fit_maxK,k_data,Pk_data,ePk,ekPk,Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,Model_type,5 )    
        if('den' in PS_type)and('mom' in PS_type)and('crs' not in PS_type): 
            Plot_denMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[:3*Nk],Pk_data[:3*Nk],ePk[:3*Nk],ekPk[:3*Nk],Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,Model_type,5  )    
            Plot_momMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[3*Nk:],Pk_data[3*Nk:],ePk[3*Nk:],ekPk[3*Nk:],Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type, Model_type,15  )    
        if('den' in PS_type)and('mom' not in PS_type)and('crs' in PS_type): 
            Plot_denMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[:3*Nk],Pk_data[:3*Nk],ePk[:3*Nk],ekPk[:3*Nk],Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,Model_type,5  )    
            Plot_crsMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[3*Nk:],Pk_data[3*Nk:],ePk[3*Nk:],ekPk[3*Nk:],Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,Model_type,15 )           
        if('den' not in PS_type)and('mom' in PS_type)and('crs' in PS_type):     
            Plot_momMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[:3*Nk],Pk_data[:3*Nk],ePk[:3*Nk],ekPk[:3*Nk],Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,Model_type,1  )    
            Plot_crsMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[3*Nk:],Pk_data[3*Nk:],ePk[3*Nk:],ekPk[3*Nk:],Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,Model_type,15 )               
        if('den' in PS_type)and('mom' in PS_type)and('crs' in PS_type): 
            Plot_denMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[:3*Nk],Pk_data[:3*Nk],ePk[:3*Nk],ekPk[:3*Nk],Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,Model_type,1  )    
            Plot_momMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[ 3*Nk:6*Nk],Pk_data[ 3*Nk:6*Nk],ePk[3*Nk:6*Nk],ekPk[3*Nk:6*Nk],Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,Model_type,6  )    
            Plot_crsMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[6*Nk:],Pk_data[6*Nk:],ePk[6*Nk:],ekPk[6*Nk:],Sig8_fid,PT,PS_type, az,Hz,Dz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,Model_type,15 )    
                          
if(strategy =='MCMC'):
    strategy = 'MCMC'  
    Optparm=param_test
    output_MCMCchian  = file_dir+'MC_'+fit_tech+'_'+PS_type+'_'+WFCon_type+'_'+PS_measure+'_'+str(fit_maxK)
    chain=  Param_Estimate( fit_minK,fit_maxK,Optparm,MCparm,PS_type,fit_tech,strategy,OnlyL0,Sig8_fid,k_obs,Pk_obs,k_mocks,Pk_mocks,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,  az,Hz,Dz, WFCon_type,Model_type)
    np.save(output_MCMCchian,chain,allow_pickle=True);
    for i in range(len(chain[0,:])):
        plt.figure(i)
        plt.hist(chain[:,i],50);
    plt.show()   
    mean  = fsig8_fid[0]
    MCplot(mean,chain,PS_type,'/Users/fei/WSP/Scie/CosmPS/FitExamp/Plots/MC'+PS_type+WFCon_type+'.png' )
 
     
