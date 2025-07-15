import numpy as np
import matplotlib.pyplot as plt
from PSmodFun import *
 


PS_measure = 'mocks' # 'survey' # 
file_dir   = '/Users/fei/WSP/Scie/FitExamp/Data/ProdPy/' 

OmegaM    = 0.3 
OmegaA    = 1.- OmegaM
hb        = 0.68
Hub       = 100.
ombh2     = 0.02210272
omch2     = 0.11661728
ns        = 0.96 
As        = 2.2154397054279834e-09
# 'As = As*sigma8_fid_sim/sigma8_fid_integ'
#KmodLim = 1.  # h/Mpc. Yes, this is a very large value for power spectrum.
#NKmodbin= 437 # the number of k-bin for PL(k) of camb.
#PT, Sig8_fid, outdir = PSloop_Fun( KmodLim,NKmodbin,Hub*hb,ombh2,omch2,ns,As,file_dir) 
#print('\n The loop terms are saved in:', outdir, '\n')
RsFeff    = 0.1
az        = 1./(1.+RsFeff)
Hz        = Hub*Ez(RsFeff, OmegaM, OmegaA, 0., -1., 0., 0.)
Dz        = Dz_Fun(RsFeff, OmegaM, OmegaA ) 





# growth rate:------------------
fsigma8        = 0.40946855 
# density field:---------------- 
b1sigma8       = 1.12568620 
b2sigma8       = 1.17890514 
bssigma8       = 'anal' 
b3nlsigma8     = 0.4335061 
sigmav1_squre  = 40.2873740 
sigmav2_squre  = 156.669623 
sigmav3_squre  = 1.8238754 
# momentum field:--------------- 
b1vsigma8      = 0.80151994 
b2vsigma8      = b2sigma8
bsvsigma8      = 'anal'
b3nlvsigma8    = 'anal'
sigmavv1_squre = 3.1344342 
sigmavv2_squre = 12.1289577
sigmavv3_squre = 1.27892345
#-------------------------------
# The list of parameters of power spectrum: 
parm = [fsigma8  ,  
        b1sigma8 , b2sigma8 , bssigma8 , b3nlsigma8   ,  
        b1vsigma8, b2vsigma8, bsvsigma8, b3nlvsigma8  ,   
        sigmav1_squre , sigmav2_squre  , sigmav3_squre,    
        sigmavv1_squre, sigmavv2_squre , sigmavv3_squre]



 

# 1. read Convmat:------------------------------------------------------------------
covrossden,kir,kjr,WF_rand=np.load(file_dir+'convden_Ross_' +PS_measure+'.npy',allow_pickle=True) 
covrossmom,kir,kjr,WF_rand=np.load(file_dir+'convmom_Ross_' +PS_measure+'.npy',allow_pickle=True) 

covden,ki,kjo=np.load(file_dir+'convden_Blake_' +PS_measure+'.npy',allow_pickle=True) 
covmom,ki,kjo=np.load(file_dir+'convmom_Blake_' +PS_measure+'.npy',allow_pickle=True) 
covcrs,ki,kjo=np.load(file_dir+'convcrs_Blake_' +PS_measure+'.npy',allow_pickle=True)
covblack=[covden,covmom,covcrs]
Nkmod=len(kjo) 
Nkdat=len(ki)

covden,kip,kjp=np.load(file_dir+'convden_Beutler_' +PS_measure+'.npy',allow_pickle=True) 
covmom,kip,kjp=np.load(file_dir+'convmom_Beutler_' +PS_measure+'.npy',allow_pickle=True) 
covcrs,kip,kjp=np.load(file_dir+'convcrs_Beutler_' +PS_measure+'.npy',allow_pickle=True)
covpypw=[covden,covmom,covcrs]
Nkdatp=len(kip)

ki=np.concatenate(np.array([list(ki),list(ki),list(ki)]))
kip=np.concatenate(np.array([list(kip),list(kip),list(kip)]))
kjb=np.array([list(kjo),list(kjo),list(kjo)])
kjp=np.array([list(kjp),list(kjp),list(kjp)])


# 3. model PS:----------------------------------------------------------------------
sst=np.load(file_dir+'INTEG_PL.npy',allow_pickle=True)
PT,Sig8_fid=sst
# 3.4 density power:-----------------
Pkmodc_denrg,kmodc_denrg,Pkmod_denrg,kmod_denrg=PkmodMulti_Fun(parm,True , Sig8_fid, kir, kjr,  covrossden , WF_rand, PT, 'den-0'  , az,Hz,Dz, 'Ross' )
Pkmodc_denrv,kmodc_denrv,Pkmod_denrv,kmod_denrv=PkmodMulti_Fun(parm,True , Sig8_fid, kir, kjr,  covrossmom , WF_rand, PT, 'den-0'  , az,Hz,Dz, 'Ross' )
Pkmodc_denb ,kmodc_denb ,Pkmod_denb ,kmod_denb =PkmodMulti_Fun(parm,False, Sig8_fid, ki , kjb,  covblack   , np.nan , PT, 'den-024', az,Hz,Dz, 'Blake' )
Pkmodc_denp ,kmodc_denp ,Pkmod_denp ,kmod_denp =PkmodMulti_Fun(parm,False, Sig8_fid, kip, kjp,  covpypw    , np.nan , PT, 'den-024', az,Hz,Dz, 'Beutler' )
# 3.5 momentum power:----------------
Pkmodc_momrg,kmodc_momrg,Pkmod_momrg,kmod_momrg=PkmodMulti_Fun(parm,True , Sig8_fid, kir, kjr,  covrossden , WF_rand, PT, 'mom-0'  , az,Hz,Dz, 'Ross' )
Pkmodc_momrv,kmodc_momrv,Pkmod_momrv,kmod_momrv=PkmodMulti_Fun(parm,True , Sig8_fid, kir, kjr,  covrossmom , WF_rand, PT, 'mom-0'  , az,Hz,Dz, 'Ross' )
Pkmodc_momb ,kmodc_momb ,Pkmod_momb ,kmod_momb =PkmodMulti_Fun(parm,False, Sig8_fid, ki , kjb,  covblack   , np.nan , PT, 'mom-024', az,Hz,Dz, 'Blake' )
Pkmodc_momp ,kmodc_momp ,Pkmod_momp ,kmod_momp =PkmodMulti_Fun(parm,False, Sig8_fid, kip, kjp,  covpypw    , np.nan , PT, 'mom-024', az,Hz,Dz, 'Beutler' )
# 3.6 cross    power:----------------
Pkmodc_crsb ,kmodc_crsb ,Pkmod_crsb ,kmod_crsb =PkmodMulti_Fun(parm,False, Sig8_fid, ki , kjb,  covblack   , np.nan , PT, 'crs-13' , az,Hz,Dz, 'Blake' )
Pkmodc_crsp ,kmodc_crsp ,Pkmod_crsp ,kmod_crsp =PkmodMulti_Fun(parm,False, Sig8_fid, kip, kjp,  covpypw    , np.nan , PT, 'crs-13' , az,Hz,Dz, 'Beutler' )



# 4 some example for PSmodMulti without any window convolution:---------------- 
Nkmod0=1000
kbin =np.linspace(0,0.4,Nkmod0)
# density ps:
Pkmodc_den0,kmodc_den0,Pkmod_den,kmod_den =PkmodMulti_Fun(parm,False, Sig8_fid, kbin, np.nan,np.nan,np.nan, PT, 'den-024', az,Hz,Dz, np.nan)
# mom ps:
Pkmodc_mom0,kmodc_mom0,Pkmod_mom,kmod_mom =PkmodMulti_Fun(parm,False, Sig8_fid, kbin, np.nan,np.nan,np.nan, PT, 'mom-024', az,Hz,Dz, np.nan)
# crs ps:
Pkmodc_crs0,kmodc_crs0,Pkmod_crs,kmod_crs =PkmodMulti_Fun(parm,False, Sig8_fid, kbin, np.nan,np.nan,np.nan, PT, 'crs-13' , az,Hz,Dz, np.nan) 




# 5. red-shift space model power spectrum:-------------------------------------
krsd,mu,Pkmodrsd_den =Pkmod_Fun(parm, Sig8_fid, PT, 'den', az,Hz,Dz,False) 
krsd,mu,Pkmodrsd_mom =Pkmod_Fun(parm, Sig8_fid, PT, 'mom', az,Hz,Dz,False) 
krsd,mu,Pkmodrsd_crs =Pkmod_Fun(parm, Sig8_fid, PT, 'crs', az,Hz,Dz,False)      



# 6. make plots:---------------------------------------------------------------
plt.figure(1,figsize=(9,6))
plt.plot(kmod_denb[0*Nkmod:1*Nkmod]  ,kmod_denb[0*Nkmod:1*Nkmod] * Pkmod_denb[0*Nkmod:1*Nkmod]  ,c='silver',lw=4, label='Model PS')
plt.plot(kmodc_den0[0*Nkmod0:1*Nkmod0]  ,kmodc_den0[0*Nkmod0:1*Nkmod0] * Pkmodc_den0[0*Nkmod0:1*Nkmod0]  ,c='k' )
plt.plot(kmodc_denrg                  ,kmodc_denrg                 * Pkmodc_denrg                  ,c='lime',label='Conv PS Ross-den')
plt.plot(kmodc_denrv                  ,kmodc_denrv                 * Pkmodc_denrv                  ,c='limegreen',label='Conv PS Ross-mom')
plt.plot(kmodc_denb[0*Nkdat:1*Nkdat] ,kmodc_denb[0*Nkdat:1*Nkdat]* Pkmodc_denb[0*Nkdat:1*Nkdat] ,c='orangered' ,  label='Conv PS Black')
plt.plot(kmodc_denp[0*Nkdatp:1*Nkdatp] ,kmodc_denp[0*Nkdatp:1*Nkdatp]* Pkmodc_denp[0*Nkdatp:1*Nkdatp] ,c='dodgerblue',label='Conv PS pypower')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('k $P^{\delta}_{0}(k)$',fontsize=23)
plt.xlabel('k [h $Mpc^{-1}$]',fontsize=23)
plt.xlim(0,0.4);#plt.ylim(0,1500)
plt.figure(2,figsize=(9,6))
plt.plot(kmod_denb[1*Nkmod:2*Nkmod]  ,kmod_denb[1*Nkmod:2*Nkmod] * Pkmod_denb[1*Nkmod:2*Nkmod]  ,c='silver',lw=4, label='Model PS')
plt.plot(kmodc_den0[1*Nkmod0:2*Nkmod0]  ,kmodc_den0[1*Nkmod0:2*Nkmod0] * Pkmodc_den0[1*Nkmod0:2*Nkmod0]  ,c='k' )
plt.plot(kmodc_denb[1*Nkdat:2*Nkdat] ,kmodc_denb[1*Nkdat:2*Nkdat]* Pkmodc_denb[1*Nkdat:2*Nkdat] ,c='orangered' ,label='Conv PS Black')
plt.plot(kmodc_denp[1*Nkdatp:2*Nkdatp] ,kmodc_denp[1*Nkdatp:2*Nkdatp]* Pkmodc_denp[1*Nkdatp:2*Nkdatp] ,c='deepskyblue',label='Conv PS pypower')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0,0.4);plt.ylim(-400,800)
plt.ylabel('k $P^{\delta}_{2}(k)$',fontsize=23)
plt.xlabel('k [h $Mpc^{-1}$]',fontsize=23)
plt.figure(3,figsize=(9,6))
plt.plot(kmod_denb[2*Nkmod:3*Nkmod]  ,kmod_denb[2*Nkmod:3*Nkmod] * Pkmod_denb[2*Nkmod:3*Nkmod]  ,c='silver',lw=4, label='Model PS')
plt.plot(kmodc_den0[2*Nkmod0:3*Nkmod0]  ,kmodc_den0[2*Nkmod0:3*Nkmod0] * Pkmodc_den0[2*Nkmod0:3*Nkmod0]  ,c='k' )
plt.plot(kmodc_denb[2*Nkdat:3*Nkdat] ,kmodc_denb[2*Nkdat:3*Nkdat]* Pkmodc_denb[2*Nkdat:3*Nkdat] ,c='orangered' ,label='Conv PS Black')
plt.plot(kmodc_denp[2*Nkdatp:3*Nkdatp] ,kmodc_denp[2*Nkdatp:3*Nkdatp]* Pkmodc_denp[2*Nkdatp:3*Nkdatp] ,c='deepskyblue',label='Conv PS pypower')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0,0.4);plt.ylim(-30,380)
plt.ylabel('k $P^{\delta}_{4}(k)$',fontsize=23)
plt.xlabel('k [h $Mpc^{-1}$]',fontsize=23)







# 7. make plots:---------------------------------------------------------------
plt.figure(4,figsize=(9,6))
plt.yscale('log')
plt.plot(kmod_momb[0*Nkmod:1*Nkmod]  , Pkmod_momb[0*Nkmod:1*Nkmod]  ,c='silver',lw=4,label='Model PS')
plt.plot(kmodc_mom0[0*Nkmod0:1*Nkmod0],Pkmodc_mom0[0*Nkmod0:1*Nkmod0]  ,c='k' )
plt.plot(kmodc_momrg                  , Pkmodc_momrg                  ,c='lime',label='Conv PS Ross-den')
plt.plot(kmodc_momrv                  , Pkmodc_momrv                  ,c='limegreen',label='Conv PS Ross-mom')
plt.plot(kmodc_momb[0*Nkdat:1*Nkdat] , Pkmodc_momb[0*Nkdat:1*Nkdat] ,c='orangered' ,label='Conv PS Black')
plt.plot(kmodc_momp[0*Nkdat:1*Nkdat] , Pkmodc_momp[0*Nkdat:1*Nkdat] ,c='dodgerblue',label='Conv PS pypower')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('$P^{p}_{0}(k)$',fontsize=23)
plt.xlabel('k [h $Mpc^{-1}$]',fontsize=23)
plt.xlim(0,0.4)
plt.figure(5,figsize=(9,6))
plt.yscale('log')
plt.plot(kmod_momb[1*Nkmod:2*Nkmod]  , Pkmod_momb[1*Nkmod:2*Nkmod]  ,c='silver',lw=4,label='Model PS')
plt.plot(kmodc_mom0[1*Nkmod0:2*Nkmod0],Pkmodc_mom0[1*Nkmod0:2*Nkmod0]  ,c='k' )
plt.plot(kmodc_momb[1*Nkdat:2*Nkdat] , Pkmodc_momb[1*Nkdat:2*Nkdat] ,c='orangered' ,label='Conv PS Black')
plt.plot(kmodc_momp[1*Nkdat:2*Nkdat] , Pkmodc_momp[1*Nkdat:2*Nkdat] ,c='dodgerblue',label='Conv PS pypower')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('$P^{p}_{2}(k)$',fontsize=23)
plt.xlabel('k [h $Mpc^{-1}$]',fontsize=23)
plt.xlim(0,0.4)
plt.figure(6,figsize=(9,6))
plt.yscale('log')
plt.plot(kmod_momb[2*Nkmod:3*Nkmod]  , Pkmod_momb[2*Nkmod:3*Nkmod]  ,c='silver',lw=4,label='Model PS')
plt.plot(kmodc_mom0[2*Nkmod0:3*Nkmod0],Pkmodc_mom0[2*Nkmod0:3*Nkmod0]  ,c='k' )
plt.plot(kmodc_momb[2*Nkdat:3*Nkdat] , Pkmodc_momb[2*Nkdat:3*Nkdat] ,c='orangered' ,label='Conv PS Black')
plt.plot(kmodc_momp[2*Nkdat:3*Nkdat] , Pkmodc_momp[2*Nkdat:3*Nkdat] ,c='dodgerblue',label='Conv PS pypower')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('$P^{p}_{4}(k)$',fontsize=23)
plt.xlabel('k [h $Mpc^{-1}$]',fontsize=23)
plt.xlim(0,0.4)







# 8. make plots:---------------------------------------------------------------
plt.figure(7,figsize=(9,6))
plt.plot(kmod_crsb[0*Nkmod:1*Nkmod]  ,kmod_crsb[0*Nkmod:1*Nkmod] * Pkmod_crsb[0*Nkmod:1*Nkmod]  ,c='silver',lw=4,label='Model PS')
plt.plot(kmodc_crs0[0*Nkmod0:1*Nkmod0]  ,kmodc_crs0[0*Nkmod0:1*Nkmod0] * Pkmodc_crs0[0*Nkmod0:1*Nkmod0]  ,c='k' )
plt.plot(kmodc_crsb[0*Nkdat:1*Nkdat] ,kmodc_crsb[0*Nkdat:1*Nkdat]* Pkmodc_crsb[0*Nkdat:1*Nkdat] ,c='orangered' ,label='Conv PS Black')
plt.plot(kmodc_crsp[0*Nkdat:1*Nkdat] ,kmodc_crsp[0*Nkdat:1*Nkdat]* Pkmodc_crsp[0*Nkdat:1*Nkdat] ,c='dodgerblue',label='Conv PS pypower')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('k $P^{\delta p}_{1}(k)$',fontsize=23)
plt.xlabel('k [h $Mpc^{-1}$]',fontsize=23)
plt.xlim(0,0.4);plt.ylim(-0.3*10**6,2*10**6)
plt.figure(8,figsize=(9,6))
plt.plot(kmod_crsb[1*Nkmod:2*Nkmod]  ,kmod_crsb[1*Nkmod:2*Nkmod] * Pkmod_crsb[1*Nkmod:2*Nkmod]  ,c='silver',lw=4,label='Model PS')
plt.plot(kmodc_crs0[1*Nkmod0:2*Nkmod0]  ,kmodc_crs0[1*Nkmod0:2*Nkmod0] * Pkmodc_crs0[1*Nkmod0:2*Nkmod0]  ,c='k' )
plt.plot(kmodc_crsb[1*Nkdat:2*Nkdat] ,kmodc_crsb[1*Nkdat:2*Nkdat]* Pkmodc_crsb[1*Nkdat:2*Nkdat] ,c='orangered' ,label='Conv PS Black')
plt.plot(kmodc_crsp[1*Nkdatp:2*Nkdatp] ,kmodc_crsp[1*Nkdatp:2*Nkdatp]* Pkmodc_crsp[1*Nkdatp:2*Nkdatp] ,c='dodgerblue',label='Conv PS pypower')
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('k $P^{\delta p}_{3}(k)$',fontsize=23)
plt.xlabel('k [h $Mpc^{-1}$]',fontsize=23)
plt.xlim(0,0.4);plt.ylim(-0.1*10**6,0.5*10**6)





#'''





