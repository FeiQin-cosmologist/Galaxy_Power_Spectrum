import numpy as np
from PSmodFun import *
from PSfitFun import * 
#==========================================================================
#                           INITIAL SETTING
#--------------------------------------------------------------------------
# 1. INITIAL SETTING:
OmegaM    = 0.3121
OmegaA    = 1.- OmegaM
Hub       = 100. 
littleh   = 0.6751
ombh2     = 0.02237
omch2     = 0.11986 
ns        = 0.9653
RsFeff    = 0.073
As        = 2.1204361365583653e-09
KmodLim   =  1.
Gamma_fid = 0.55
 

PS_type    = 'den-02 mom-02 crs-1'
PS_measure = 'mockAve'#  'survey'#        'mockML'    #           
WFCon_type = 'Blake'    # 'Ross'   #   'Beutler'#        is only for l=0, if using multiple, you have to use 'Black' .
OnlyL0     =  False     #  True #               # only fit l=0
strategy   = 'Optimize' #'MCMC'  #      
fit_tech   =  'BC'    #    'ZP' # 'Chi2'     #           
IntegCal   =  False   # True # 
fit_minK   =  0.025       
fit_maxK   =  0.3  
Nmock      =  599
MCparm     = [500,3000]
Optparm    = [0.463297,1.12900,1.23989,-0.37794,0.99779,64.17674,160.61045,38.10959 ] 

file_dir   = '/Users/fei/WSP/Scie/CosmPS/Data/ProdPy/'



#==============================================================================
#                                   MAIN CODE
# 1. read data:---------------------------------------------------------------- 
if(WFCon_type=='Ross' ): OnlyL0=True ;  
Nk,k_data,Pk_data,k_mock,Pk_mock,conv, kmodc,wf_rand=LoadPS_Fun(file_dir,Nmock,PS_type,WFCon_type,OnlyL0,PS_measure)  
Pave,   ePk,  ekPk =  PSmock_prop(k_mock,Pk_mock,fit_minK ,fit_maxK,PS_measure )
Growthz = GrowthFactorGR(RsFeff, OmegaM, OmegaA, 0.0, 100.0, -1.0, 0.0, 0.0)/GrowthFactorGR(0.0, OmegaM, OmegaA, 0.0, 100.0, -1.0, 0.0, 0.0)
if(PS_measure =='mockAve')or(PS_measure =='mockML'): Pk_data= Pave
print(PS_measure)

# 2. Calculate the integrations:-----------------------------------------------
if(IntegCal):PT,Sig8_fid,outdir=PSloop_Fun( KmodLim,Hub*littleh,ombh2,omch2,ns,As,file_dir) 
if(not IntegCal):
    sst=np.load(file_dir+'INTEG_Pl.npy',allow_pickle=True)
    PT,Sig8_fid=sst
print('Sig8_fid =',Sig8_fid,'\n') 
fsig8_fid=Fsigma8_Fun(Gamma_fid,RsFeff,Sig8_fid, OmegaM, OmegaA,  Hub)
print('fsig8_fid=',fsig8_fid) 
print('The above fsig8_fid is not the fiducial of sim box, the sim box is in z=0 rather than z=z_eff=',RsFeff)
print('So, the fiducial fsig8 of sim box should simply euqal to OmegaM^0.55*Sig8_fid=',OmegaM**Gamma_fid*Sig8_fid,'\n')

# 3. Fit parms:----------------------------------------------------------------
k_obs        = k_data
Pk_obs       = Pk_data
k_mocks      = k_mock
Pk_mocks     = Pk_mock 
kmod_conv    = kmodc
WF_Kobs_Kmod = conv
WF_rand      = wf_rand
  
if(strategy =='Optimize'):
    fsigma8,  b1sigma8,b2sigma8,b3nlsigma8,  b1vsigma8, sigmav1_squre,sigmav2_squre,sigmavv_squre,  chi2s= Param_Estimate( fit_minK,fit_maxK,Optparm,MCparm,PS_type,fit_tech,strategy,OnlyL0,Sig8_fid,k_obs,Pk_obs,k_mocks,Pk_mocks,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,  Growthz, WFCon_type)
    print( 'fsigma8   =', fsigma8)
    print( 'b12sigma8 =', b1sigma8,b2sigma8,b3nlsigma8 )
    print( 'b1vsigma8 =', b1vsigma8  )
    print( 'sigmav^2  =', sigmav1_squre,sigmav2_squre,sigmavv_squre  ) 
    print( 'CHI2      =', chi2s   )    
    param_test = [ fsigma8,  b1sigma8,b2sigma8,b3nlsigma8,  b1vsigma8, sigmav1_squre,sigmav2_squre,sigmavv_squre   ]  
    if(OnlyL0)    : Plot_l0(param_test , Nk,fit_minK,fit_maxK,Sig8_fid, k_obs, Pk_obs,ekPk,ePk, kmod_conv, WF_Kobs_Kmod,WF_rand,PT,PS_type, Growthz, WFCon_type,0 )  
    if(not OnlyL0): 
        if('den' in PS_type)and('mom' not in PS_type)and('crs' not in PS_type):
            Plot_denMulti(param_test ,Nk,fit_minK,fit_maxK,k_data,Pk_data,ePk,ekPk,Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,5 )    
        if('mom' in PS_type)and('den' not in PS_type)and('crs' not in PS_type):
            Plot_momMulti(param_test ,Nk,fit_minK,fit_maxK,k_data,Pk_data,ePk,ekPk,Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,5 )    
        if('crs' in PS_type)and('den' not in PS_type)and('mom' not in PS_type):
            Plot_crsMulti(param_test ,Nk,fit_minK,fit_maxK,k_data,Pk_data,ePk,ekPk,Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,5 )    
        if('den' in PS_type)and('mom' in PS_type)and('crs' not in PS_type): 
            Plot_denMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[:3*Nk],Pk_data[:3*Nk],ePk[:3*Nk],ekPk[:3*Nk],Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,5  )    
            Plot_momMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[3*Nk:],Pk_data[3*Nk:],ePk[3*Nk:],ekPk[3*Nk:],Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type, 15  )    
        if('den' in PS_type)and('mom' not in PS_type)and('crs' in PS_type): 
            Plot_denMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[:3*Nk],Pk_data[:3*Nk],ePk[:3*Nk],ekPk[:3*Nk],Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,5  )    
            Plot_crsMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[3*Nk:],Pk_data[3*Nk:],ePk[3*Nk:],ekPk[3*Nk:],Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,15 )           
        if('den' not in PS_type)and('mom' in PS_type)and('crs' in PS_type):     
            Plot_momMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[:3*Nk],Pk_data[:3*Nk],ePk[:3*Nk],ekPk[:3*Nk],Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,1  )    
            Plot_crsMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[3*Nk:],Pk_data[3*Nk:],ePk[3*Nk:],ekPk[3*Nk:],Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,15 )               
        if('den' in PS_type)and('mom' in PS_type)and('crs' in PS_type): 
            Plot_denMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[:3*Nk],Pk_data[:3*Nk],ePk[:3*Nk],ekPk[:3*Nk],Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,1  )    
            Plot_momMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[ 3*Nk:6*Nk],Pk_data[ 3*Nk:6*Nk],ePk[3*Nk:6*Nk],ekPk[3*Nk:6*Nk],Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type,6  )    
            Plot_crsMulti(param_test ,Nk,fit_minK,fit_maxK,k_data[6*Nk:],Pk_data[6*Nk:],ePk[6*Nk:],ekPk[6*Nk:],Sig8_fid,PT,PS_type, Growthz,kmod_conv,WF_Kobs_Kmod,WF_rand, WFCon_type ,15 )    
                          
if(strategy =='MCMC'):
    strategy = 'MCMC'   
    Optparm=param_test
    output_MCMCchian  = file_dir+'MC_'+fit_tech+'_'+PS_type+'_'+WFCon_type+'_'+PS_measure+'_'+str(fit_maxK)
    chain=  Param_Estimate( fit_minK,fit_maxK,Optparm,MCparm,PS_type,fit_tech,strategy,OnlyL0,Sig8_fid,k_obs,Pk_obs,k_mocks,Pk_mocks,kmod_conv, WF_Kobs_Kmod,WF_rand,PT,  Growthz, WFCon_type)
    np.save(output_MCMCchian,chain,allow_pickle=True);
    for i in range(len(chain[0,:])):
        plt.figure(i)
        plt.hist(chain[:,i],50);
    plt.show()   
    mean  = [0.432,-100 ]  
    MCplot(mean,chain,'/Users/fei/WSP/Scie/Proj12/Results/MC'+PS_type+WFCon_type+'.png' )
 

 

     
        
        
