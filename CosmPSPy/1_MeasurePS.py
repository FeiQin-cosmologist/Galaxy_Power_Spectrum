import numpy as np
from PSestFun import *
###############################################################################
#####                                                                     #####
#####                           Initial Settings                          #####
#####                                                                     #####
###############################################################################
# fiducial cosmological parameters: 
OmegaM          = 0.3121
OmegaA          = 1.-OmegaM
Hub             = 100.
sigv            = 300.0
# power spectrum parameters
FKP             = 1600.
FKPv            = 5.*10.**9
PS_multi        ='all'  #  'no' # 'yes' # 
Nmock           = 599
Nrand           = 1
# grid and k-bin for mocks and data:
kmin,kmax,nk    = 0.0,0.4,40
nx,ny,nz        = 376,376,376 
xmin,xmax       = -376. ,376. 
ymin,ymax       = -376. ,376.
zmin,zmax       = -376. ,376. 
# grid and k-bin for randoms:
kminr,kmaxr,nkr = 0.0,0.4,240
nxr,nyr,nzr     = 356,356,356 
xminr,xmaxr     = -356. ,356. 
yminr,ymaxr     = -356. ,356.
zminr,zmaxr     = -356. ,356.
# redshift bins for generateing vp of randoms:
NczBin          = 50
# Observer's frame infomation:
Bulk_vel        = np.array([0.,0.,0.])
Bulk_velmk      = np.array([0.,0.,0.])
# data file directory:
Input_dir  = '/Users/fei/WSP/Scie/CosmPS/Data/Orig/'  
Output_dir = '/Users/fei/WSP/Scie/CosmPS/Data/ProdPy/'





















###############################################################################
#####                                                                     #####
#####                           power spectrum                            #####
#####                                                                     #####
###############################################################################
print('')
print('Nmock=',Nmock)
print('Nrand=',Nrand)
print('')
for icod in range(6):    
    #icod = 30;   PS_type = 'crs' ; Do_mock= False;     
    iRUN = 0  
    if(icod==0): PS_type = 'mom' ; Do_mock= False  
    if(icod==1): PS_type = 'mom' ; Do_mock= True  
    if(icod==2): PS_type = 'den' ; Do_mock= False    
    if(icod==3): PS_type = 'den' ; Do_mock= True  
    if(icod==4): PS_type = 'crs' ; Do_mock= False    
    if(icod==5): PS_type = 'crs' ; Do_mock= True  
    if( not Do_mock): 
        print('')
        print('  real survey')  
#==============================================================================        
# 1. density power spectrum of galaxy data and randoms: 
        if((PS_type=='den')or(PS_type=='crs')):
# 1.1: read galaxy survey data: 
            infile   = np.load(Input_dir+'Data.npy',allow_pickle=True)
            ra       = infile[0]   
            dec      = infile[1]
            cz       = infile[2] 
            nb       = infile[3]
            ndata    = len(ra)
# 1.2: reand Randoms of galaxy survey:
            infiler  = np.load(Input_dir+'Rand.npy',allow_pickle=True)
            raR      = infiler[0]
            decR     = infiler[1]
            czR      = infiler[2] 
            nbR      = infiler[3] 
            ndataR   = len(raR)  
            if(Nrand==1):NBRmk = infiler[3]
# 1.3: calculate density power spectrum of galaxy survey: 
            if(PS_type=='den'):
                out_dir,wfkp, Norm = Pkest_Fun(kmin,kmax,nk,nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,
                                               ra,dec,cz/LightSpeed,nb,FKP,raR,decR,czR/LightSpeed,nbR,
                                               np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                                               OmegaM,OmegaA,Hub,sigv, 
                                               bulk_vel=np.nan,PS_type='den',PS_multi=PS_multi,file_dir=Output_dir+'Data_den' ) 
# 1.4: calculate momentum power spectrum of random data: 
                out_dir,wfkpR = PkestR_Fun(kminr,kmaxr,nkr,nxr,nyr,nzr,xminr,xmaxr,yminr,ymaxr,zminr,zmaxr,ndata,
                                           raR,decR,czR/LightSpeed,np.nan,nbR,FKP,
                                           OmegaM,OmegaA,Hub,sigv,'den',Output_dir+'Rand_den') 
# 1.5: ratio between randoms and galaxies: 
# you need these values for window function convolution if using Beutler method:   
                print('galaxy survey: Norm=',Norm)
                print('galaxy survey: Nrand/Ndata  =',ndataR/ndata)
                print('galaxy survey: Nrandw/Ndataw=',np.sum(wfkpR)/np.sum(wfkp[0]))  
                
#==============================================================================        
# 2. momentum power spectrum of velocity data and randoms: 
        if((PS_type=='mom')or(PS_type=='crs')):             
# 2.1 read velocity survey data:
            infile   = np.load(Input_dir+'Datav.npy',allow_pickle=True)
            rav      = infile[0]   
            decv     = infile[1]
            czv      = infile[2] 
            logd     = infile[3]
            elogd    = infile[4]
            nbv      = infile[5]  
            ndatav    = len(rav)
            vp        = logd_to_Vp(logd,czv/LightSpeed,OmegaM) # watkins2015 estimator. 
            evp       = czv/(1.+czv/LightSpeed )*np.log(10.)*elogd  
# 2.2 Randoms for velocity survey data: 
            infiler  = np.load(Input_dir+'Randv.npy',allow_pickle=True)
            ravR     = infiler[0]
            decvR    = infiler[1]
            czvR     = infiler[2] 
            nbvR     = infiler[3] 
            vpR,evpR = VpRand_Fun(logd,czv,czvR,NczBin,OmegaM) 
            ndatavR  = len(ravR)
# 2.3: calculate momentum power spectrum of velocity survey: 
            if(PS_type=='mom'):
                out_dir,wfkpv, Normv = Pkest_Fun(kmin,kmax,nk,nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,
                                                 np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                                                 rav,decv,czv/LightSpeed,vp,evp,nbv,FKPv,
                                                 OmegaM,OmegaA,Hub,sigv,
                                                 bulk_vel=Bulk_vel,PS_type='mom',PS_multi=PS_multi,file_dir=Output_dir+'Data_mom'  )
# 2.4: calculate momentum power spectrum of random data: 
                out_dir,wfkpvR = PkestR_Fun(kminr,kmaxr,nkr,nxr,nyr,nzr,xminr,xmaxr,yminr,ymaxr,zminr,zmaxr,ndatav,
                                            ravR,decvR,czvR/LightSpeed,evpR,nbvR,FKPv,
                                            OmegaM,OmegaA,Hub,sigv,'mom',Output_dir+'Rand_mom') 
            # 2.5: ratio between randoms and galaxies: 
                print('velocity survey: Norm=',Normv)
                print('velocity survey: Nrand/Ndata=',ndatavR/ndatav)
                print('velocity survey: Nrandw/Ndataw=',np.sum(wfkpvR)/np.sum(wfkpv[0]))
                
#==============================================================================        
# 3. cross power spectrum of real survey: 
        if(PS_type=='crs'):    
            out_dir,wfkpc,Normc = Pkest_Fun(kmin,kmax,nk,nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,
                                            ra ,dec ,cz /LightSpeed ,nb,FKP,  
                                            raR,decR,czR/LightSpeed ,nbR, 
                                            rav,decv,czv/LightSpeed ,vp,evp,nbv,FKPv,   
                                            OmegaM,OmegaA,Hub,sigv,
                                             bulk_vel=Bulk_vel,PS_type='crs',PS_multi=PS_multi,file_dir=Output_dir+'Data_crs')
            print('cross survey: Norm=',Normc)
                
#==============================================================================        
    if(Do_mock):
        print('')
        print('  mocks')
        print('')
        #if you have more than one random catalogue
        ndatamk=[] ;wfkpmk=[] ;Normmk=[]
        ndatavmk=[];wfkpvmk=[];Normvmk=[];Normcmk=[]    
        if((Nrand == 1)and(PS_type!='mom')):
            raRmk,decRmk,czRmk ,nbRmk,ndataRmk=raR,decR,czR ,nbR,ndataR
        else: ndataRmk=[]    
        ndatavRmk=ndatavR;
        #-------------------------------------------
        for i_mockS in range(Nmock):
            i_mock=i_mockS+Nmock*iRUN     
            print('  i_mock =',i_mock)
#==============================================================================         
# 4. density power spectrum of mocks: 
            if((PS_type=='den')or(PS_type=='crs')):
# 4.1: read galaxy survey data: 
                infile   = np.load(Input_dir+'Mock_'+str(i_mock)+'.npy',allow_pickle=True)
                ra       = infile[0]   
                dec      = infile[1]
                cz       = infile[2] 
                nb       = infile[3]  
                ndatamk.append( len(ra)  )
                if(Nrand>1):
                    infiler  = np.load(Input_dir+'MockRand_'+str(i_mock)+'.npy',allow_pickle=True)
                    raRmk      = infiler[0]
                    decRmk     = infiler[1]
                    czRmk      = infiler[2] 
                    nbRmk      = infiler[3]
                    ndataRmk.append(len(raRmk))  
# 4.2: calculate density power spectrum: 
                if(PS_type=='den'):
                    if( Nrand == 1): nbRmk=NBRmk/(len(ra)/len(raRmk)) * ndata/len(raRmk) 
                    out_dir,tmpwfkp,tmpNorm = Pkest_Fun(kmin,kmax,nk,nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,
                                                   ra,dec,cz/LightSpeed,nb,FKP,raRmk,decRmk,czRmk/LightSpeed ,nbRmk,
                                                   np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                                                   OmegaM,OmegaA,Hub,sigv, 
                                                   bulk_vel=np.nan,PS_type='den',PS_multi=PS_multi,file_dir=Output_dir+'Mock_den'+str(i_mock) ) 
                    wfkpmk.append(np.array( np.sum(wfkpR)/np.sum(tmpwfkp[0]) )) ;Normmk.append(tmpNorm)
#==============================================================================        
# 5. momentum power spectrum of mocks: 
            if((PS_type=='mom')or(PS_type=='crs')):             
# 5.1 read velocity survey data:
                infile   = np.load(Input_dir+'Mockv_'+str(i_mock)+'.npy',allow_pickle=True)
                rav      = infile[0]   
                decv     = infile[1]
                czv      = infile[2] 
                logd     = infile[3]
                elogd    = infile[4]
                nbv      = infile[5]   
                ndatavmk.append( len(rav))
                vp       = logd_to_Vp(logd,czv/LightSpeed,OmegaM)  
                evp      = czv/(1.+czv/LightSpeed )*np.log(10.)*elogd  
# 5.2: calculate momentum power spectrum of velocity survey: 
                if(PS_type=='mom'):
                    out_dir,tmpwfkpv, tmpNormv = Pkest_Fun(kmin,kmax,nk,nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,
                                                 np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                                                 rav,decv,czv/LightSpeed,vp,evp,nbv,FKPv,
                                                 OmegaM,OmegaA,Hub,sigv,
                                                 bulk_vel=Bulk_velmk,PS_type='mom',PS_multi=PS_multi,file_dir=Output_dir+'Mock_mom'+str(i_mock)  )
                    wfkpvmk.append(np.array( np.sum(wfkpvR)/np.sum(tmpwfkpv[0]) )) ;Normvmk.append(tmpNormv)
#==============================================================================        
# 6. cross power spectrum of mocks:
                if(PS_type=='crs'): 
                    if( Nrand == 1): nbRmk=NBRmk/(len(ra)/len(raRmk)) * ndata/len(raRmk) 
                    out_dir,wfkpc, tmpNormc = Pkest_Fun(kmin,kmax,nk,nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax,
                                                    ra ,dec ,cz /LightSpeed ,nb,FKP,  
                                                    raRmk,decRmk,czRmk/LightSpeed ,nbRmk, 
                                                    rav,decv,czv/LightSpeed ,vp,evp,nbv,FKPv,   
                                                    OmegaM,OmegaA,Hub,sigv,
                                                    bulk_vel=Bulk_velmk,PS_type='crs',PS_multi=PS_multi,file_dir=Output_dir+'Mock_crs'+str(i_mock))
                    Normcmk.append(tmpNormc)
#==============================================================================        
# 7. some out puts: 
        if(PS_type=='den'):        
            ndatamk=np.array(ndatamk) ;wfkpmk=np.array(wfkpmk) ;Normmk=np.array(Normmk) ;
            print('galaxy mocks: Norm=',np.mean(Normmk))
            print('galaxy mocks: Nrand/Ndata=',np.mean(ndataRmk/ndatamk))
            print('galaxy mocks: Nrandw/Ndataw=',np.mean(wfkpmk))
        if(PS_type=='mom'):
            ndatavmk=np.array(ndatavmk) ;wfkpvmk=np.array(wfkpvmk) ;Normvmk=np.array(Normvmk) ;
            print('velocity mocks: Norm=',np.mean(Normvmk))
            print('velocity mocks: Nrand/Ndata=',np.mean(ndatavRmk/ndatavmk))
            print('velocity mocks: Nrandw/Ndataw=',np.mean(wfkpvmk))
        if(PS_type=='crs'):    
            Normcmk=np.array(Normcmk) ;
            print('cross mocks: Norm=',np.mean(Normcmk))
#====================            The end           ============================





     