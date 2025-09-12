import numpy as np
from PSformatData import *


 


# fiducial cosmological parameters: 
OmegaM          = 0.3 
OmegaA          = 1.-OmegaM
Hub             = 100.
sigv            = 300.0
# power spectrum parameters
FKP             = 1600.
FKPv            = 5.*10.**9
Nmock           = 599
# redshift bins for generateing vp of randoms:
NczBin          = 50
# data file directory:
Input_dir  = '/Users/fei/WSP/Scie/FitExamp/Data/Orig/'  
Output_dir = '/Users/fei/WSP/Scie/FitExamp/Data/ProdFt/'
 









print('')
print('Nmock=',Nmock)
print('')
# 1: galaxy survey:------------------------------------------------------------ 
infile   = np.load(Input_dir+'Data.npy',allow_pickle=True)
ra       = infile[0]   
dec      = infile[1]
cz       = infile[2] 
nb       = infile[3] 
ndata    = len(ra)
outdir_galsurvey,wfkp = FormatData_Fun([ra,dec,cz,nb,FKP],Output_dir+'Data', 'gal-survey')
# Randoms: 
infiler  = np.load(Input_dir+'Rand.npy',allow_pickle=True)
raR      = infiler[0]
decR     = infiler[1]
czR      = infiler[2] 
nbR      = infiler[3] 
ndataR   = len(raR)   
# formating data: 
outdir_galsurveyrand,wfkpR = FormatData_Fun([raR,decR,czR,     nbR*ndata/ndataR,    FKP], Output_dir+'Rand', 'gal-rand')
print('galaxy survey: Nrand/Ndata  =',ndataR/ndata)
print('galaxy survey: Nrandw/Ndataw=',np.sum(wfkpR)/np.sum(wfkp))

 

# 2: PV survey:---------------------------------------------------------------- 
infile   = np.load(Input_dir+'Datav.npy',allow_pickle=True)
rav      = infile[0]   
decv     = infile[1]
czv      = infile[2] 
logd     = infile[3]
elogd    = infile[4]
nbv      = infile[5] 
ndatav   = len(rav)
outdir_pvsurvey,wfkpv=FormatData_Fun([rav,decv,czv,logd,elogd,nbv, sigv,FKP, FKPv],Output_dir+'Datav', 'pv-survey')
# Randoms for survey data: 
infiler  = np.load(Input_dir+'Randv.npy',allow_pickle=True)
ravR     = infiler[0]
decvR    = infiler[1]
czvR     = infiler[2] 
nbvR     = infiler[3] 
vpR,evpR = VpRand_Fun(logd,czv,czvR,NczBin,OmegaM) 
ndatavR  = len(ravR) 
# formating data:  
outdir_pvsurveyrand,wfkpvR=FormatData_Fun([ravR,decvR,czvR,vpR,evpR,    nbvR*ndatav/ndatavR,    sigv,FKP, FKPv], Output_dir+'Randv', 'pv-rand')
print('velocity survey: Nrand/Ndata=',ndatavR/ndatav)
print('velocity survey: Nrandw/Ndataw, den-mom=',np.sum(wfkpvR[0])/np.sum(wfkpv[0]),np.sum(wfkpvR[1])/np.sum(wfkpv[1]))

  

# 3: galaxy Mocks:------------------------------------------------------------- 
ndatamk=[];wfkp=[]
for i_mock in range(Nmock):
    # mocks:
    infile   = np.load(Input_dir+'Mock_'+str(i_mock)+'.npy',allow_pickle=True)
    ra       = infile[0]   
    dec      = infile[1]
    cz       = infile[2] 
    nb       = infile[3] 
    ndata_mk    = len(ra)
    ndatamk.append( ndata_mk   ) 
    outdir_galmock,ss=FormatData_Fun([ra,dec,cz,nb,FKP],Output_dir+'Mock_'+str(i_mock) , 'gal-mock')
    wfkp.append(np.sum(wfkpR)/np.sum(ss))
print('galaxy mocks: Nrand/Ndata  =',np.mean(ndataR/np.array(ndatamk)))     
print('galaxy mocks: Nrandw/Ndataw=',np.mean(wfkp))



# 4: pv Mocks:----------------------------------------------------------------- 
ndatavmk=[];wfkp1=[];wfkp2=[]
for i_mock in range(Nmock):
    # mocks:
    infile   = np.load(Input_dir+'Mockv_'+str(i_mock)+'.npy',allow_pickle=True)
    rav      = infile[0]   
    decv     = infile[1]
    czv      = infile[2] 
    logd     = infile[3]
    elogd    = infile[4]
    nbv      = infile[5]  
    ndatavmk.append( len(rav)  ) 
    outdir_pvmock,ss=FormatData_Fun([rav,decv,czv,logd,elogd,nbv, sigv,FKP, FKPv],Output_dir+'Mockv_'+str(i_mock) , 'pv-mock')
    wfkp1.append(np.sum(wfkpvR[0])/np.sum(ss[0]))
    wfkp2.append(np.sum(wfkpvR[1])/np.sum(ss[1]))
print('velocity mocks: Nrand/Ndata=',np.mean(ndatavR/np.array(ndatavmk)))    
print('velocity mocks: Nrandw/Ndataw, den-mom=',np.mean(wfkp1),np.mean(wfkp2))


#------------------------------------------------------------------------------
print(outdir_galsurvey)
print(outdir_galsurveyrand)
print(outdir_galmock)
#print(outdir_galmockrand)
print(outdir_pvsurvey)
print(outdir_pvsurveyrand)
print(outdir_pvmock)
#print(outdir_pvmockrand)



 


 