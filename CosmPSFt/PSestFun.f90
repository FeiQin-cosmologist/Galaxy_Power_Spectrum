module parm_grid_kbin
implicit none
!==============================================================================
!                              INTRODUCTION
!------------------------------------------------------------------------------
! kmin, kmax, nkw: the minimum and max values of the wave number k.  'nkw' is the number of k-bins in [kmin, kmax]. 
! The galaxies are grided in a 3D-cube (x,y,z). 
! nx, ny, nz: the number of grid in x,y,z directions. 
! XMIN, XMAX: the minimum and max values of x Mpc/h
! YMIN, YMAX: the minimum and max values of y Mpc/h
! ZMIN, ZMAX: the minimum and max values of z Mpc/h
! The above 12 variables are the only things you need to modify in this code. Do not modify the rest of the code. 
!#####################     Setting Grid parameters     ########################


real(8),parameter    ::   kmin=0.      ,   kmax=0.4


!momPS
integer(8),parameter ::   nx=376, ny=376, nz=376, nkw=40
real(8),parameter    ::   XMIN=-376.   ,   XMAX=376.!mpc/h
real(8),parameter    ::   YMIN=-376.   ,   YMAX=376.!mpc/h
real(8),parameter    ::   ZMIN=-376.   ,   ZMAX=376.!mpc/h



!#############################     
end module parm_grid_kbin!####
module parm_grid_kbin_rand!###
implicit none!################
!############################# 


!grid and kbin parm for the mom and den PS of random
real(8),parameter    ::   kmin=0.      ,   kmax=0.4 
integer(8),parameter ::   nx=356, ny=356, nz=356, nkw=240
real(8),parameter    ::   XMIN=-356.   ,   XMAX=356.!mpc/h
real(8),parameter    ::   YMIN=-356.   ,   YMAX=356.!mpc/h
real(8),parameter    ::   ZMIN=-356.   ,   ZMAX=356.!mpc/h


!#############################    The    End    ###############################
































































































 

! Do not change the following code:
!==============================================================================
end module parm_grid_kbin_rand
module momPS_interface
interface
    subroutine cosmic_PS(do_mom ,PS_Multi,output_dir,   &
                                   ndatapv, nbmom,  Xpv,Ypv,Zpv,Vpec,wmom,wrho,&                                                                 
                                   ndata ,  nb   ,   X,  Y,  Z,   w ,  &
                                   ndataR,  nbr,    Xr, Yr, Zr, wr ) 
    integer(8),intent(in) ::   do_mom
    integer(8),intent(in)::ndata,ndataR,ndatapv
    real(8),intent(in),allocatable,dimension(:):: Xpv,Ypv,Zpv,X,Y,Z,Vpec,w,wr,wmom,wrho
    real(8),intent(in),allocatable,dimension(:):: Xr,Yr,Zr,nb,nbr,nbmom
    character(1000),intent(in)::output_dir
    character(10),intent(in):: PS_Multi 
    end subroutine cosmic_PS
end interface
end module momPS_interface
!--------------------------    subroutine.2    ----------------------------
subroutine   cosmic_PS(do_mom ,PS_Multi,output_dir,&
                                   ndatapv, nbmom,  Xpv,Ypv,Zpv,Vpec,wmom,wrho,&                                                                 
                                   ndata ,  nb   ,   X,  Y,  Z,   w ,  &
                                   ndataR,  nbr,    Xr, Yr, Zr, wr ) 
use parm_grid_kbin
implicit none
real(8)   ,parameter ::   Pi=3.1415926535897932
include "fftw3.f"
!----------------------------------------
integer(8),intent(in) ::   do_mom
integer(8),intent(in)::ndata,ndataR,ndatapv
real(8),intent(in),allocatable,dimension(:):: Xpv,Ypv,Zpv,X,Y,Z,Vpec,w,wr,wmom
real(8),intent(in),allocatable,dimension(:):: Xr,Yr,Zr,nb,nbr,nbmom,wrho
character(1000),intent(in)::output_dir
character(10),intent(in):: PS_Multi
!----------------------------------------
integer(8)::ipart,ix,iy,iz,i,j,k,ikw,ii,jj,kk,jusm
real(8),allocatable,dimension(:)::Vp
real(8) :: dx,dy,dz,Pvnoise,dk,PnoiseC
real(8) :: fx_nqu,fy_nqu,fz_nqu,min_nqu,Pnoise,Norm,PnoiseR,NormR
real(8),allocatable,dimension(:,:,:)::FUN,FUN_save,FUNmom,FUN_savemom
integer(8),parameter:: nzh=(nz/2)+1
double complex FUN_OUT( nzh,nx, ny)
double complex FUN_OUT_save( nzh,nx, ny)
double complex FUN_outmom( nzh,nx, ny)
double complex FUN_out_savemom( nzh,nx, ny)
integer*8 plan_forward
integer*8 plan_forward2
real(8):: fx,fy,fz,kw,sincx,sincy,sincz,grid_cor,Lvect,kprefac,alpha
real(8),allocatable,dimension(:)::pg0,pg1,pg2,pg3,pg4,Pg0shot,Pg2shot
integer(8),allocatable,dimension(:)::NbinKW
real(8),dimension(3)::vect,kvect

!1.--------------------------------------
! settings
if((do_mom == 1) .or. (do_mom == 2)) print*,'     N pv   =',ndatapv
if((do_mom == 0) .or. (do_mom == 2)) print*,'     N gal  =',ndata
if((do_mom == 1) .or. (do_mom == 2)) then
    allocate(vp(ndatapv))
    vp = vpec
endif
 

 

!2.--------------------------------------
!将各个星系放入到相应的格子中: grid the galaxies
allocate(FUN(NZ,NX,NY))
dx  = (XMAX-XMIN)/NX;   dy  = (YMAX-YMIN)/NY;   dz  = (ZMAX-ZMIN)/NZ;
FUN = 0.
!assign galaxies to grids:
if((do_mom == 0) .or. (do_mom == 2))then
  DO ipart=1,ndata
    ix=(x(ipart)-XMIN)/dx; iy=(y(ipart)-YMIN)/dy;iz=(z(ipart)-ZMIN)/dz;
    if(ix<0)  ix=ix+Nx;      if(iy<0)  iy=iy+ny;     if(iz<0)  iz=iz+NZ;
    if(ix>=nx)ix=ix-Nx;      if(iy>=ny)iy=iy-ny;     if(iz>=nz)iz=iz-NZ;
    FUN(iz,ix,iy) = FUN(iz,ix,iy) + w(ipart)
  ENDDO
endif  
if(do_mom == 2)then
    allocate(FUNmom(NZ,NX,NY))
    FUNmom=0.
endif    
if((do_mom == 1) .or. (do_mom == 2))then
  DO ipart=1,ndatapv
    ix=(xpv(ipart)-XMIN)/dx; iy=(ypv(ipart)-YMIN)/dy;iz=(zpv(ipart)-ZMIN)/dz;
    if(ix<0)  ix=ix+Nx;      if(iy<0)  iy=iy+ny;     if(iz<0)  iz=iz+NZ;
    if(ix>=nx)ix=ix-Nx;      if(iy>=ny)iy=iy-ny;     if(iz>=nz)iz=iz-NZ;
    if(do_mom == 1)FUN(iz,ix,iy) = FUN(iz,ix,iy) + wmom(ipart)*vp(ipart)
    if(do_mom == 2)FUNmom(iz,ix,iy) = FUNmom(iz,ix,iy) + wmom(ipart)*vp(ipart)
  ENDDO
endif  
!assign random points to grids and minus the random from FUN fro density power:
IF ((do_mom == 0) .or. (do_mom == 2))then
    alpha=sum(w)/sum(wr)
    DO ipart=1,ndataR
        ix=(xr(ipart)-XMIN)/dx; iy=(yr(ipart)-YMIN)/dy
        iz=(zr(ipart)-ZMIN)/dz;
        if(ix<0)  ix=ix+Nx;    if(iy<0)  iy=iy+ny;   if(iz<0)  iz=iz+NZ;
        if(ix>=nx)ix=ix-Nx;    if(iy>=ny)iy=iy-ny;   if(iz>=nz)iz=iz-NZ;
        FUN(iz,ix,iy)=FUN(iz,ix,iy)-alpha*wr(ipart)
    ENDDO
ENDIF

 


!3.--------------------------------------
!求归一化常数和噪音项: normalization factor of PS
if (do_mom==1)then
    Pvnoise=sum(wmom*wmom*vp*vp)
    Norm   = sum(nbmom*wmom*wmom)
endif    
if(do_mom == 0)then
    Pnoise = sum(w*w)!nbw2
    PnoiseR= alpha*alpha*sum(wr*wr)    
    Norm   = sum(nb*w*w)!nb2w2
    !NormR  = alpha*sum(nbr*wr*wr)
endif
if (do_mom==2)then
    PnoiseC= sum(wrho*wmom*vp)
    Norm = sqrt( sum(nb*w*w))*sqrt(sum(nbmom*wmom*wmom) ) 
endif     



! 4.-------------------------------------
!奈奎斯特频率 Nyquist frequency,f_nqu
!只有远小于f_nqu才能获得不失真的信号,或者波长大于格子长度的波才不失真.
fx_nqu=PI/dx  ;  fy_nqu=PI/dy  ;  fz_nqu=PI/dz  ;  min_nqu=fx_nqu;
if(fy_nqu<min_nqu) min_nqu=fy_nqu;
if(fz_nqu<min_nqu) min_nqu=fz_nqu;
!奈奎斯特波长等于格子尺寸的一半.






! 5.-------------------------------------
!对每个格子中的FUN做傅里叶变换,FUN共有NX*NY*NZ
!个元素, 变换以后得到NX*NY*(NZ/2+1)个元素.
! this is to calcuate the l=0 power.
call dfftw_plan_dft_r2c_3d_ (plan_forward,nx,ny,nz,FUN,FUN_out,FFTW_ESTIMATE )
call dfftw_execute_ ( plan_forward )
if(do_mom==2)then
    call dfftw_plan_dft_r2c_3d_ (plan_forward2,nx,ny,nz,FUNmom,FUN_outmom,FFTW_ESTIMATE )
    call dfftw_execute_ ( plan_forward2 )
endif



 

!6.--------------------------------------
!计算功率谱的两个模: l=0  power
dk=(kmax-kmin)/nkw
allocate(pg0(nkw));allocate(NbinKW(nkw))
if((PS_Multi=='YES').or.(PS_Multi=='ALL'))then
   allocate(pg2(nkw));pg2=0.
   allocate(pg4(nkw));pg4=0.
   allocate(pg1(nkw));pg1=0.
   allocate(pg3(nkw));pg3=0.   
endif
nbinkw=0
pg0=0.
DO j=1,nx
    !---------------------------
    !以下这段程序求出给定格子与坐标原
    !点之间相距多少个格子,以及距离对
    !的应的波数.the wave vectors
    if(j<=Nx/2)then
        fx = (j-1)/(Nx*dx)
    else
        fx = (j-1-Nx)/(Nx*dx)
    endif
    DO k=1,ny
       if(k<=Ny/2)then
           fy = (k-1)/(Ny*dy)
       else
           fy = (k-1-Ny)/(Ny*dy)
       endif
       DO i=1,nz/2
           fz = (i-1)/(Nz*dz)
           kw=2.*pi*sqrt(fx*fx+fy*fy+fz*fz)
           ikw=(kw-kmin)/dk
    !---------------------------
           IF( (ikw>=0).and.(ikw<nkw).and.(kw<(0.5*min_nqu)) )then
              !定义辛克函数:window function convolution correction
              sincx=1.;sincy=1.;sincz=1.
              if(fx .ne. 0.) sincx = sin(fx*dx*pi)/(fx*dx*pi);
              if(fy .ne. 0.) sincy = sin(fy*dy*pi)/(fy*dy*pi);
              if(fz .ne. 0.) sincz = sin(fz*dz*pi)/(fz*dz*pi);
              grid_cor=1./(sincx*sincy*sincz);
              !求A0*A0:
              if (do_mom==0)then
                 Pg0(ikw+1) =Pg0(ikw+1)+ (real(FUN_out(i,j,k))**2+aimag(FUN_out(i,j,k))**2  - (Pnoise+PnoiseR))&
                 *grid_cor*grid_cor
              endif
              if (do_mom==1)then
                 Pg0(ikw+1) =Pg0(ikw+1)+ (real(FUN_out(i,j,k))**2+aimag(FUN_out(i,j,k))**2  - Pvnoise)*grid_cor*grid_cor
              endif
              if ((do_mom==2).and.(PS_Multi=='ALL'))then ! the cross power is 0 when l=0
                 Pg0(ikw+1) = Pg0(ikw+1)+ ( 0.5*( real(FUN_out(i,j,k))     * aimag(FUN_outmom(i,j,k))  &
                                                    -aimag(FUN_out(i,j,k))    * real(FUN_outmom(i,j,k))   &
                                                    +real(FUN_outmom(i,j,k))  * aimag(FUN_out(i,j,k))     &
                                                    -aimag(FUN_outmom(i,j,k)) * real(FUN_out(i,j,k))      &
                                                   )                                                      &
                                        - PnoiseC) * grid_cor*grid_cor !power = (dkr*dki_mom-dki*dkr_mom)*grid_cor;
              endif                             
              nbinkw(ikw+1)=nbinkw(ikw+1)+1
              if((PS_Multi=='YES').or.(PS_Multi=='ALL'))then! in this step FUN_out is A0, so we are calculate A0*A0
                  if(do_mom .le. 1)then
                      pg2(ikw+1) =Pg2(ikw+1)-1./2.*(real(FUN_out(i,j,k))**2+aimag(FUN_out(i,j,k))**2)*grid_cor*grid_cor
                      pg4(ikw+1) =Pg4(ikw+1)+0.375*(real(FUN_out(i,j,k))**2+aimag(FUN_out(i,j,k))**2)*grid_cor*grid_cor
                  endif
                  if((do_mom .eq. 2).and.(PS_Multi=='ALL'))then  ! the cross power is 0 when l=0
                      pg2(ikw+1) =Pg2(ikw+1)-1./2.* 0.5*( real(FUN_out(i,j,k))     * aimag(FUN_outmom(i,j,k))  &
                                                         -aimag(FUN_out(i,j,k))    * real(FUN_outmom(i,j,k))   &
                                                         +real(FUN_outmom(i,j,k))  * aimag(FUN_out(i,j,k))     &
                                                         -aimag(FUN_outmom(i,j,k)) * real(FUN_out(i,j,k))      &
                                                        ) * grid_cor*grid_cor                                                                
                      pg4(ikw+1) =Pg4(ikw+1)+0.375* 0.5*( real(FUN_out(i,j,k))     * aimag(FUN_outmom(i,j,k))  &
                                                         -aimag(FUN_out(i,j,k))    * real(FUN_outmom(i,j,k))   &
                                                         +real(FUN_outmom(i,j,k))  * aimag(FUN_out(i,j,k))     &
                                                         -aimag(FUN_outmom(i,j,k)) * real(FUN_out(i,j,k))      &
                                                        ) * grid_cor*grid_cor          
                  endif                                                                              
              endif
           ENDIF
       ENDDO
   ENDDO
ENDDO
allocate(FUN_save(NZ,NX,NY))
FUN_save=FUN ! FUN_save is the original density field function without FFT
FUN=0.
FUN_out_save=FUN_out !  A0rho:  A0 for density field rho
if(do_mom .eq. 2)then 
    allocate(FUN_savemom(NZ,NX,NY))
    FUN_savemom=FUNmom ! FUN_savemom is the original momentum field function without FFT
    FUNmom=0.
    FUN_out_savemom=FUN_outmom ! A0p:  A0 for momentum field p 
endif



 







!7.------------------------------------
!calculate power spectrum multiples: 
!============
IF(PS_Multi=='NO')goto 133
IF((PS_Multi=='YES').or.(PS_Multi=='ALL'))THEN


!============
! 7.1: calculate l=1: -----
DO ii=1,3 
! calculate the field function projected to l=1 modes:
If( (do_mom .le. 1).and.(PS_Multi=='ALL'))then
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN  /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUN(iz,ix,iy) = FUN_save(iz,ix,iy)*vect(ii)/sqrt(Lvect) ! FUN is the original l=1 field function without FFT
      enddo
   enddo
enddo
endif
if(do_mom==2)then
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN  /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUN(iz,ix,iy) = FUN_save(iz,ix,iy)*vect(ii)/sqrt(Lvect) ! FUN is the original l=1 field function without FFT
      enddo
   enddo
enddo
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN  /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUNmom(iz,ix,iy) = FUN_savemom(iz,ix,iy)*vect(ii)/sqrt(Lvect) ! FUN is the original l=1 field function without FFT
      enddo
   enddo
enddo
endif
! calculate FFT
If( (do_mom .le. 1).and.(PS_Multi=='ALL')) then
    call dfftw_execute_ ( plan_forward )
endif
if(do_mom==2)then
    call dfftw_execute_ ( plan_forward )
    call dfftw_execute_ ( plan_forward2 )
endif
!7.3 calculate Pg1:---
Do j=1,nx
   if(j<=Nx/2)then
      fx = (j-1)/(Nx*dx)
   else
      fx = (j-1-Nx)/(Nx*dx)
   endif
   Do k=1,ny
      if(k<=Ny/2)then
         fy = (k-1)/(Ny*dy)
      else
         fy = (k-1-Ny)/(Ny*dy)
      endif
      Do i=1,nz/2
         fz = (i-1)/(Nz*dz)
         kw=2.*pi*sqrt(fx*fx+fy*fy+fz*fz)
         Kvect=(/ 2.*pi*fz , 2.*pi*fx, 2.*pi*fy /)
         ikw=(kw-kmin)/dk
         !---------------------------
         IF( (ikw>=0).and.(ikw<nkw).and.(kw<(0.5*min_nqu)) )then
            !定义辛克函数:
            sincx=1.;sincy=1.;sincz=1.
            if(fx .ne. 0.) sincx = sin(fx*dx*pi)/(fx*dx*pi);
            if(fy .ne. 0.) sincy = sin(fy*dy*pi)/(fy*dy*pi);
            if(fz .ne. 0.) sincz = sin(fz*dz*pi)/(fz*dz*pi);
            grid_cor=1./(sincx*sincy*sincz);
            !求A1*A0:
            kprefac = 1.0;  
            If( (do_mom .le. 1).and.(PS_Multi=='ALL'))then
                if (kw > 0.0)then
                  Pg1(ikw+1) = Pg1(ikw+1)+  1.0*kprefac* kvect(ii) /kw                &
                             *  (  real(FUN_out_save(i,j,k))  * aimag(FUN_out(i,j,k)) & 
                                  -aimag(FUN_out_save(i,j,k)) * real(FUN_out(i,j,k))  &                              
                                 )* grid_cor*grid_cor                                 
                  Pg3(ikw+1) = Pg3(ikw+1)-  1.5*kprefac* kvect(ii)/kw                 &
                             *  (  real(FUN_out_save(i,j,k))  * aimag(FUN_out(i,j,k)) & 
                                  -aimag(FUN_out_save(i,j,k)) * real(FUN_out(i,j,k))  &                                
                                 )* grid_cor*grid_cor                
                endif                      
            Endif     
            If(do_mom .eq. 2)then    
                if (kw > 0.0)then
                  Pg1(ikw+1) = Pg1(ikw+1)+  1.0*kprefac* kvect(ii) /kw        &
                             * 0.5*(-aimag(FUN_out(i,j,k))    * real(FUN_out_savemom(i,j,k))   &
                                    +real(FUN_out(i,j,k))     * aimag(FUN_out_savemom(i,j,k))  &                                  
                                    +aimag(FUN_outmom(i,j,k)) * real(FUN_out_save(i,j,k))      & 
                                    -real(FUN_outmom(i,j,k))  * aimag(FUN_out_save(i,j,k))     & 
                                   )* grid_cor*grid_cor                                 
                  Pg3(ikw+1) = Pg3(ikw+1)-  1.5*kprefac* kvect(ii)/kw         &
                             * 0.5*(-aimag(FUN_out(i,j,k))    * real(FUN_out_savemom(i,j,k))   &
                                    +real(FUN_out(i,j,k))     * aimag(FUN_out_savemom(i,j,k))  &                                  
                                    +aimag(FUN_outmom(i,j,k)) * real(FUN_out_save(i,j,k))      & 
                                    -real(FUN_outmom(i,j,k))  * aimag(FUN_out_save(i,j,k))     & 
                                   )* grid_cor*grid_cor                                                                                                                                                             
                endif
             Endif   
         ENDIF
         !---------------------------
      ENDDo
   ENDDo
ENDDo
!=============
ENDDO
If( (do_mom .le. 1).and.(PS_Multi=='ALL')) FUN=0.
If(do_mom .eq. 2)then 
FUNmom=0. 
FUN=0.
endif
 

 


!=============
! 7.2: calculate l=2
DO ii=1,3
DO jj=ii,3
! FUN is the original l=2 field function without FFT
if(do_mom .le. 1)then
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN   /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUN(iz,ix,iy) = FUN_save(iz,ix,iy)*vect(ii)*vect(jj)/Lvect ! FUN is the original l=2 field function without FFT
      enddo
   enddo
enddo
endif
if((do_mom==2).and.(PS_Multi=='ALL'))then
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN  /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUN(iz,ix,iy) = FUN_save(iz,ix,iy)*vect(ii)*vect(jj)/Lvect ! FUN is the original l=2 field function without FFT
      enddo
   enddo
enddo
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN   /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUNmom(iz,ix,iy) = FUN_savemom(iz,ix,iy)*vect(ii)*vect(jj)/Lvect ! FUN is the original l=2 field function without FFT
      enddo
   enddo
enddo
endif
!  傅里叶变换:--------
!call dfftw_plan_dft_r2c_3d_ (plan_forward,nx,ny,nz,FUN,FUN_out,FFTW_ESTIMATE )
if(do_mom .le. 1)then
    call dfftw_execute_ ( plan_forward )
endif    
if((do_mom==2).and.(PS_Multi=='ALL'))then
    call dfftw_execute_ ( plan_forward )
    call dfftw_execute_ ( plan_forward2 )
endif
! calculate Pg2:---
Do j=1,nx
   if(j<=Nx/2)then
      fx = (j-1)/(Nx*dx)
   else
      fx = (j-1-Nx)/(Nx*dx)
   endif
   Do k=1,ny
      if(k<=Ny/2)then
         fy = (k-1)/(Ny*dy)
      else
         fy = (k-1-Ny)/(Ny*dy)
      endif
      Do i=1,nz/2
         fz = (i-1)/(Nz*dz)
         kw=2.*pi*sqrt(fx*fx+fy*fy+fz*fz)
         Kvect=(/ 2.*pi*fz , 2.*pi*fx, 2.*pi*fy /)
         ikw=(kw-kmin)/dk
         !---------------------------
         IF( (ikw>=0).and.(ikw<nkw).and.(kw<(0.5*min_nqu)) )then
            !定义辛克函数:
            sincx=1.;sincy=1.;sincz=1.
            if(fx .ne. 0.) sincx = sin(fx*dx*pi)/(fx*dx*pi);
            if(fy .ne. 0.) sincy = sin(fy*dy*pi)/(fy*dy*pi);
            if(fz .ne. 0.) sincz = sin(fz*dz*pi)/(fz*dz*pi);
            grid_cor=1./(sincx*sincy*sincz);
            !求A2*A0:
            kprefac = 1.0;
            if (ii .ne. jj)kprefac = 2.0;
            If (do_mom .le. 1)then
               if (kw > 0.0)then
                  Pg2(ikw+1) = Pg2(ikw+1)+  1.5*kprefac  &  ! A0*A2
                             * kvect(ii)*kvect(jj)/kw**2         &
                             * (real(FUN_out(i,j,k))             & ! A2=FUN_out,  A0=FUN_out_save
                             * real(FUN_out_save(i,j,k))         &
                             + aimag(FUN_out(i,j,k))             &
                             * aimag(FUN_out_save(i,j,k)) )      &
                             * grid_cor*grid_cor
                  Pg4(ikw+1) = Pg4(ikw+1)-  3.75*kprefac  & ! A0*A2
                             * kvect(ii)*kvect(jj)/kw**2         &
                             * (real(FUN_out(i,j,k))             &
                             * real(FUN_out_save(i,j,k))         &
                             + aimag(FUN_out(i,j,k))             &
                             * aimag(FUN_out_save(i,j,k)) )      &
                             * grid_cor*grid_cor                                                                  
               endif
            Endif
            If((do_mom .eq. 2).and.(PS_Multi=='ALL'))then
               if (kw > 0.0)then
                  Pg2(ikw+1) =  Pg2(ikw+1)+  1.5*kprefac  &
                             * kvect(ii)*kvect(jj)/kw**2         &
                             * 0.5*( real(FUN_out(i,j,k))     * aimag(FUN_out_savemom(i,j,k))  &
                                    -aimag(FUN_out(i,j,k))    * real(FUN_out_savemom(i,j,k))   &
                                    +real(FUN_outmom(i,j,k))  * aimag(FUN_out_save(i,j,k))     &
                                    -aimag(FUN_outmom(i,j,k)) * real(FUN_out_save(i,j,k))      &
                                   )* grid_cor*grid_cor
                  Pg4(ikw+1) = Pg4(ikw+1)-  3.75*kprefac  &  
                             * kvect(ii)*kvect(jj)/kw**2         &
                             * 0.5*( real(FUN_out(i,j,k))     * aimag(FUN_out_savemom(i,j,k))  &
                                    -aimag(FUN_out(i,j,k))    * real(FUN_out_savemom(i,j,k))   &
                                    +real(FUN_outmom(i,j,k))  * aimag(FUN_out_save(i,j,k))     &
                                    -aimag(FUN_outmom(i,j,k)) * real(FUN_out_save(i,j,k))      &
                                   )* grid_cor*grid_cor                              
               endif
            Endif
         ENDIF
         !---------------------------
      ENDDo
   ENDDo
ENDDo
!===== 
ENDDO
ENDDO
!=====
FUN=0.
If(do_mom .eq. 2)FUNmom=0. 




 



!============================
! 7.3 calculate l=3:
jusm=0
DO ii=1,3 
DO jj=1,3
DO kk=1,3 
if((jusm .ne. 3) .and. (jusm .ne. 4) .and. (jusm .ne. 6) .and. (jusm .ne. 7) .and. (jusm .ne. 8))then
if((jusm .ne. 9) .and. (jusm .ne. 10) .and. (jusm .ne. 11) .and. (jusm .ne. 15) .and. (jusm .ne. 16))then
if((jusm .ne. 17) .and. (jusm .ne. 18) .and. (jusm .ne. 19) .and. (jusm .ne. 20) .and. (jusm .ne. 21))then
if((jusm .ne. 22) .and. (jusm .ne. 23))then
!----------------------------
If( (do_mom .le. 1).and.(PS_Multi=='ALL'))then
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN   /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUN(iz,ix,iy) = FUN_save(iz,ix,iy)*vect(ii)*vect(jj)*vect(kk)/(Lvect*sqrt(Lvect) ) ! FUN is the original l=1 field function without FFT
      enddo
   enddo
enddo
endif
if(do_mom==2)then
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN   /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUN(iz,ix,iy) = FUN_save(iz,ix,iy)*vect(ii)*vect(jj)*vect(kk)/(Lvect*sqrt(Lvect) ) ! FUN is the original l=1 field function without FFT
      enddo
   enddo
enddo
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN   /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUNmom(iz,ix,iy) = FUN_savemom(iz,ix,iy)*vect(ii)*vect(jj)*vect(kk)/(Lvect*sqrt(Lvect) ) ! FUN is the original l=1 field function without FFT
      enddo
   enddo
enddo
endif
! calculate FFT
If( (do_mom .le. 1).and.(PS_Multi=='ALL')) then
    call dfftw_execute_ ( plan_forward )
endif    
if(do_mom==2)then
    call dfftw_execute_ ( plan_forward )
    call dfftw_execute_ ( plan_forward2 )
endif
!  calculate Pg3:---
Do j=1,nx
   if(j<=Nx/2)then
      fx = (j-1)/(Nx*dx)
   else
      fx = (j-1-Nx)/(Nx*dx)
   endif
   Do k=1,ny
      if(k<=Ny/2)then
         fy = (k-1)/(Ny*dy)
      else
         fy = (k-1-Ny)/(Ny*dy)
      endif
      Do i=1,nz/2
         fz = (i-1)/(Nz*dz)
         kw=2.*pi*sqrt(fx*fx+fy*fy+fz*fz)
         Kvect=(/ 2.*pi*fz , 2.*pi*fx, 2.*pi*fy /)
         ikw=(kw-kmin)/dk
         !---------------------------
         IF( (ikw>=0).and.(ikw<nkw).and.(kw<(0.5*min_nqu)) )then
            !定义辛克函数:
            sincx=1.;sincy=1.;sincz=1.
            if(fx .ne. 0.) sincx = sin(fx*dx*pi)/(fx*dx*pi);
            if(fy .ne. 0.) sincy = sin(fy*dy*pi)/(fy*dy*pi);
            if(fz .ne. 0.) sincz = sin(fz*dz*pi)/(fz*dz*pi);
            grid_cor=1./(sincx*sincy*sincz);
            !求A3*A0:
            kprefac = 1.0;  
            if (ii .ne. kk) then
                if (ii .ne. jj) then
                  kprefac = 6.0;
                else
                  kprefac = 3.0;
                endif
            endif                                   
            If( (do_mom .le. 1).and.(PS_Multi=='ALL'))then
               if (kw > 0.0)then
                  Pg3(ikw+1) = Pg3(ikw+1)+ 2.5*kprefac* kvect(ii)* kvect(jj)* kvect(kk)/kw**3  &
                             *  (  real(FUN_out_save(i,j,k))  * aimag(FUN_out(i,j,k))       & 
                                  -aimag(FUN_out_save(i,j,k)) * real(FUN_out(i,j,k))        &                                  
                                 )* grid_cor*grid_cor                                           
               endif
            Endif                     
            If (do_mom .eq. 2)then         
               if (kw > 0.0)then                                
                  Pg3(ikw+1) = Pg3(ikw+1)+ 2.5*kprefac* kvect(ii)* kvect(jj)* kvect(kk)/kw**3     &
                             * 0.5*(-aimag(FUN_out(i,j,k))    * real(FUN_out_savemom(i,j,k))   &
                                    +real(FUN_out(i,j,k))     * aimag(FUN_out_savemom(i,j,k))  &                                  
                                    +aimag(FUN_outmom(i,j,k)) * real(FUN_out_save(i,j,k))      & 
                                    -real(FUN_outmom(i,j,k))  * aimag(FUN_out_save(i,j,k))     & 
                                   )* grid_cor*grid_cor                               
               endif
            Endif   
         ENDIF
         !---------------------------
      ENDDo
   ENDDo
ENDDo
endif
endif
endif
endif
jusm=jusm+1
ENDDO
ENDDO
ENDDO
If( (do_mom .le. 1).and.(PS_Multi=='ALL')) FUN=0.
If(do_mom .eq. 2)then
    FUNmom=0. 
    FUN=0.
endif

 























!8.------------------------------------
!calculate l=4
!============
jusm=1
DO ii=1,3 
DO jj=1,3
DO kk=1,3 
if((jusm .ne. 4)  .and.  (jusm .ne. 7)  .and. (jusm .ne. 8)  .and. (jusm .ne. 10)) then 
if((jusm .ne. 11) .and.  (jusm .ne. 16) .and. (jusm .ne. 17) .and. (jusm .ne. 19)) then
if((jusm .ne. 21) .and.  (jusm .ne. 22) .and. (jusm .ne. 23) .and. (jusm .ne. 24)) then 
!============
!8.1 计算4极子项: ! the original field FUN_save projected to ri*rj*rk to obtain FUN(l=4)
if(do_mom .le. 1)then
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN   /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUN(iz,ix,iy) = FUN_save(iz,ix,iy)*vect(ii)*vect(ii)*vect(jj)*vect(kk)/Lvect**2! FUN is the original l=4 field function without FFT
      enddo
   enddo
enddo
endif
if((do_mom==2).and.(PS_Multi=='ALL'))then
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN   /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUN(iz,ix,iy) = FUN_save(iz,ix,iy)*vect(ii)*vect(ii)*vect(jj)*vect(kk)/Lvect**2! FUN is the original l=4 field function without FFT
      enddo
   enddo
enddo
do ix=1,NX
   do iy=1,NY
      do iz=1,NZ
         vect =(/  iz*dz+ZMIN  , &
                   ix*dx+XMIN  , &
                   iy*dy+YMIN   /)
         Lvect=vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3)
         if (Lvect == 0.0)Lvect = 1.0;
         FUNmom(iz,ix,iy) = FUN_savemom(iz,ix,iy)*vect(ii)*vect(ii)*vect(jj)*vect(kk)/Lvect**2! FUN is the original l=4 field function without FFT
      enddo
   enddo
enddo
endif
!8.2 傅里叶变换:--------
!call dfftw_plan_dft_r2c_3d_ (plan_forward,nx,ny,nz,FUN,FUN_out,FFTW_ESTIMATE )
if(do_mom .le. 1)then
    call dfftw_execute_ ( plan_forward )
endif    
if((do_mom==2).and.(PS_Multi=='ALL'))then
    call dfftw_execute_ ( plan_forward )
    call dfftw_execute_ ( plan_forward2 )
endif
!8.3 calculate Pg2:---
Do j=1,nx
   if(j<=Nx/2)then
      fx = (j-1)/(Nx*dx)
   else
      fx = (j-1-Nx)/(Nx*dx)
   endif
   Do k=1,ny
      if(k<=Ny/2)then
         fy = (k-1)/(Ny*dy)
      else
         fy = (k-1-Ny)/(Ny*dy)
      endif
      Do i=1,nz/2
         fz = (i-1)/(Nz*dz)
         kw=2.*pi*sqrt(fx*fx+fy*fy+fz*fz)
         Kvect=(/ 2.*pi*fz , 2.*pi*fx, 2.*pi*fy /)
         ikw=(kw-kmin)/dk
         !---------------------------
         IF( (ikw>=0).and.(ikw<nkw).and.(kw<(0.5*min_nqu)) )then
            !定义辛克函数:
            sincx=1.;sincy=1.;sincz=1.
            if(fx .ne. 0.) sincx = sin(fx*dx*pi)/(fx*dx*pi);
            if(fy .ne. 0.) sincy = sin(fy*dy*pi)/(fy*dy*pi);
            if(fz .ne. 0.) sincz = sin(fz*dz*pi)/(fz*dz*pi);
            grid_cor=1./(sincx*sincy*sincz);
            !求 A4*A0:          
            kprefac = 1.0;
            if (ii .eq. jj) then
                if (ii .ne. kk) kprefac = 4.0;
            else 
                if (jj .eq. kk)then
                  kprefac = 6.0;
                else  
                  kprefac = 12.0;
                endif
            endif           
            If (do_mom .le. 1)then
               if (kw > 0.0)then
                  Pg4(ikw+1) = Pg4(ikw+1)+  4.375*kprefac            & ! A4*A0
                             * kvect(ii)*kvect(ii)*kvect(jj)*kvect(kk)/kw**4 &
                             * (real(FUN_out(i,j,k))                         & ! A4=FUN_out,  A0=FUN_out_save
                             * real(FUN_out_save(i,j,k))                     &
                             + aimag(FUN_out(i,j,k))                         &
                             * aimag(FUN_out_save(i,j,k)) )                  &
                             * grid_cor*grid_cor
               endif            
            Endif
            If ((do_mom .eq. 2).and.(PS_Multi=='ALL'))then
               if (kw > 0.0)then            
                  Pg4(ikw+1) =  Pg4(ikw+1)+  4.375*kprefac            &
                             * kvect(ii)*kvect(ii)*kvect(jj)*kvect(kk)/kw**4 &
                             * 0.5*( real(FUN_out(i,j,k))     * aimag(FUN_out_savemom(i,j,k))  &
                                    -aimag(FUN_out(i,j,k))    * real(FUN_out_savemom(i,j,k))   &
                                    +real(FUN_outmom(i,j,k))  * aimag(FUN_out_save(i,j,k))     &
                                    -aimag(FUN_outmom(i,j,k)) * real(FUN_out_save(i,j,k))      &
                                   )* grid_cor*grid_cor        
               endif
            Endif
         ENDIF
         !---------------------------
      ENDDo
   ENDDo
ENDDo            
!==== 
endif
endif
endif
jusm=jusm+1
ENDDO
ENDDO
ENDDO    
!====


ENDIF
!============
133  IF(PS_Multi=='NO') print*,'     Only calulate l=0.'




!! This is the shortnoise multipole for Density PS: 
!!real(8):: Lfct
!!real(8),allocatable,dimension(:)::Pg0shot,Pg1shot,Pg2shot,Pg3shot,Pg4shot
!print*,'This is the shortnoise multipole for Density PS:'
!allocate(Pg0shot(nkw));allocate(Pg1shot(nkw));allocate(Pg2shot(nkw));
!allocate(Pg3shot(nkw));allocate(Pg4shot(nkw));
!Pg0shot=0.
!Pg1shot=0.
!Pg2shot=0.
!Pg3shot=0.
!Pg4shot=0.
!IF((PS_Multi=='YES').or.(PS_Multi=='ALL'))then
!Do j=1,nx
!   if(j<=Nx/2)then
!      fx = (j-1)/(Nx*dx)
!   else
!      fx = (j-1-Nx)/(Nx*dx)
!   endif
!   Do k=1,ny
!      if(k<=Ny/2)then
!         fy = (k-1)/(Ny*dy)
!      else
!         fy = (k-1-Ny)/(Ny*dy)
!      endif
!      Do i=1,nz/2
!         fz = (i-1)/(Nz*dz)
!         kw=2.*pi*sqrt(fx*fx+fy*fy+fz*fz)
!         Kvect=(/ 2.*pi*fz , 2.*pi*fx, 2.*pi*fy /)
!         ikw=(kw-kmin)/dk
!         IF( (ikw>=0).and.(ikw<nkw).and.(kw<(0.5*min_nqu)) )then
!            sincx=1.;sincy=1.;sincz=1.
!            if(fx .ne. 0.) sincx = sin(fx*dx*pi)/(fx*dx*pi);
!            if(fy .ne. 0.) sincy = sin(fy*dy*pi)/(fy*dy*pi);
!            if(fz .ne. 0.) sincz = sin(fz*dz*pi)/(fz*dz*pi);
!            grid_cor=1./(sincx*sincy*sincz);
!            do ii=1,ndata
!               if(kw>0.)then
!                 Lfct = (Kvect(2)*x(ii)+Kvect(3)*y(ii)+Kvect(1)*z(ii)) &
!                        / ( kw* sqrt(x(ii)**2 +y(ii)**2 + z(ii)**2)  )                 
!                 Pg0shot(ikw+1)=Pg0shot(ikw+1) + w(ii)*w(ii)         *grid_cor*grid_cor;
!                 Pg1shot(ikw+1)=Pg1shot(ikw+1) + w(ii)*w(ii)*Lfct    *grid_cor*grid_cor;
!                 Pg2shot(ikw+1)=Pg2shot(ikw+1) + w(ii)*w(ii)*Lfct**2 *grid_cor*grid_cor;
!                 Pg3shot(ikw+1)=Pg3shot(ikw+1) + w(ii)*w(ii)*Lfct**3 *grid_cor*grid_cor;
!                 Pg4shot(ikw+1)=Pg4shot(ikw+1) + w(ii)*w(ii)*Lfct**4 *grid_cor*grid_cor;
!               endif
!            enddo         
!            do ii=1,ndatar
!               Lfct = (Kvect(2)*xr(ii)+Kvect(3)*yr(ii)+Kvect(1)*zr(ii)) &
!                      / ( kw* sqrt(xr(ii)**2 +yr(ii)**2 + zr(ii)**2)  ) 
!               Pg0shot(ikw+1)=Pg0shot(ikw+1) + alpha*alpha*wr(ii)*wr(ii)          *grid_cor*grid_cor;
!               Pg1shot(ikw+1)=Pg1shot(ikw+1) + alpha*alpha*wr(ii)*wr(ii)*Lfct     *grid_cor*grid_cor;
!               Pg2shot(ikw+1)=Pg2shot(ikw+1) + alpha*alpha*wr(ii)*wr(ii)*Lfct**2 *grid_cor*grid_cor;
!               Pg3shot(ikw+1)=Pg3shot(ikw+1) + alpha*alpha*wr(ii)*wr(ii)*Lfct**3 *grid_cor*grid_cor;
!               Pg4shot(ikw+1)=Pg4shot(ikw+1) + alpha*alpha*wr(ii)*wr(ii)*Lfct**4 *grid_cor*grid_cor;
!            enddo
!         ENDIF
!      ENDDo
!   ENDDo
!ENDDo
!ENDIF
!do ikw=1,Nkw
!    if( nbinkw(ikw)>0. ) then
!        Pg0(ikw) = Pg0(ikw)/(nbinkw(ikw)*Norm);
!        if((PS_Multi=='YES').or.(PS_Multi=='ALL'))then
!           Pg1(ikw) = (Pg1(ikw)- (      Pg1shot(ikw)                                        )    )*3./(nbinkw(ikw)*Norm);
!           Pg2(ikw) = (Pg2(ikw)- (1.5  *Pg2shot(ikw) - 0.5 *Pg0shot(ikw)                     )    )*5./(nbinkw(ikw)*Norm);
!           Pg3(ikw) = (Pg3(ikw)- (2.5  *Pg3shot(ikw) - 1.5 *Pg1shot(ikw)                     )    )*7./(nbinkw(ikw)*Norm);
!           Pg4(ikw) = (Pg4(ikw)- (4.375*Pg4shot(ikw) - 3.75*Pg2shot(ikw) + 0.375*Pg0shot(ikw)  )    )*9./(nbinkw(ikw)*Norm);
!        endif
!    endif
!enddo
!!deallocate(Pg0shot);deallocate(Pg1shot);
!!deallocate(Pg2shot);deallocate(Pg3shot);deallocate(Pg4shot);





!9.------------------------------------
!归一化:
do ikw=1,Nkw
    if( nbinkw(ikw)>0. ) then
        Pg0(ikw) = Pg0(ikw)/(nbinkw(ikw)*Norm);
        if((PS_Multi=='YES').or.(PS_Multi=='ALL'))then
           Pg1(ikw) = Pg1(ikw)*3./(nbinkw(ikw)*Norm);
           Pg2(ikw) = Pg2(ikw)*5./(nbinkw(ikw)*Norm);
           Pg3(ikw) = Pg3(ikw)*7./(nbinkw(ikw)*Norm);
           Pg4(ikw) = Pg4(ikw)*9./(nbinkw(ikw)*Norm);
        endif
    endif
enddo
!将最后一个bin设置为零:
nbinkw(nkw)=0.
!Pg0(nkw)=0.
!输出文件:
open (unit=401123,file=output_dir,form='formatted')
if((PS_Multi=='YES').or.(PS_Multi=='ALL'))then
  do i=1,nkw
    if(i==nkw)then
        write(401123,*)i, real(kmin+((i-1)+0.5)*dk),0,0,0,0,0,0,0
    else
        write(401123,*)i, real(kmin+((i-1)+0.5)*dk),real(Pg0(i)),real(Pg1(i)),real(pg2(i)),real(Pg3(i)),real(pg4(i)),nbinkw(i),Norm
    endif
  enddo
endif
!--------------
if(PS_Multi=='NO')then
  do i=1,nkw
    if(i==nkw)then
      write(401123,*)i, real(kmin+((i-1)+0.5)*dk),0,0,0,0,0,0,0
    else
      write(401123,*)i, real(kmin+((i-1)+0.5)*dk),real(Pg0(i)),0,0,0,0,nbinkw(i),Norm
    endif
  enddo
endif
close(4401123)
!---------------------
deallocate(FUN);deallocate(FUN_save);deallocate(pg0);deallocate(NbinKW);
if(do_mom ==2)then
    deallocate(FUNmom);deallocate(FUN_savemom);deallocate(vp) 
endif
if(do_mom ==1)deallocate(vp)
if((PS_Multi=='YES').or.(PS_Multi=='ALL'))then
    deallocate(pg2);deallocate(pg4);deallocate(pg1);deallocate(pg3)
endif 
!the end.
end subroutine cosmic_PS



























!#########################################################################

!                             PS of random

!#########################################################################



module randPS_interface
interface
    subroutine rand_PS(do_mom ,ndataR,nbr,Xr,Yr,Zr,wr,output_dir)
    integer(8),intent(in) ::   do_mom
    integer(8),intent(in)::ndataR
    real(8),intent(in),allocatable,dimension(:):: wr
    real(8),intent(in),allocatable,dimension(:):: Xr,Yr,Zr,nbr
    character(1000),intent(in)::output_dir
    end subroutine rand_PS
end interface
end module randPS_interface

subroutine rand_PS(do_mom,ndataR,nbr,Xr,Yr,Zr,wr,output_dir)
use parm_grid_kbin_rand
implicit none
real(8)   ,parameter ::   Pi=3.1415926535897932
include "fftw3.f"
!----------------------------------------
integer(8),intent(in) ::   do_mom
integer(8),intent(in)::ndataR
real(8),intent(in),allocatable,dimension(:):: wr
real(8),intent(in),allocatable,dimension(:):: Xr,Yr,Zr,nbr
character(1000),intent(in)::output_dir
!----------------------------------------
integer(8)::ipart,ix,iy,iz,i,j,k,ikw,ii,jj
real(8) :: dx,dy,dz,dk
real(8) :: fx_nqu,fy_nqu,fz_nqu,min_nqu,PnoiseR,NormR
real(8),allocatable,dimension(:,:,:)::FUN
integer(8),parameter:: nzh=(nz/2)+1
double complex FUN_OUT( nzh,nx, ny)
integer*8 plan_forward
real(8):: fx,fy,fz,kw,sincx,sincy,sincz,grid_cor
real(8),allocatable,dimension(:)::pg0
integer(8),allocatable,dimension(:)::NbinKW
 




!1.--------------------------------------
!将各个星系放入到相应的格子中:
allocate(FUN(NZ,NX,NY))
dx  = (XMAX-XMIN)/NX;   dy  = (YMAX-YMIN)/NY;   dz  = (ZMAX-ZMIN)/NZ;
FUN = 0.
DO ipart=1,ndataR
    ix=(xr(ipart)-XMIN)/dx; iy=(yr(ipart)-YMIN)/dy
    iz=(zr(ipart)-ZMIN)/dz;
    if(ix<0)  ix=ix+Nx;    if(iy<0)  iy=iy+ny;   if(iz<0)  iz=iz+NZ;
    if(ix>=nx)ix=ix-Nx;    if(iy>=ny)iy=iy-ny;   if(iz>=nz)iz=iz-NZ;
    FUN(iz,ix,iy)=FUN(iz,ix,iy)+wr(ipart)
ENDDO
!求噪音项:
PnoiseR= sum(wr*wr)







! 2.-------------------------------------
!奈奎斯特频率
fx_nqu=PI/dx  ;  fy_nqu=PI/dy  ;  fz_nqu=PI/dz  ;  min_nqu=fx_nqu;
if(fy_nqu<min_nqu) min_nqu=fy_nqu;
if(fz_nqu<min_nqu) min_nqu=fz_nqu;
!对每个格子中的FUN做傅里叶变换
call dfftw_plan_dft_r2c_3d_ (plan_forward,nx,ny,nz,FUN,FUN_out,FFTW_ESTIMATE )
call dfftw_execute_ ( plan_forward )







!3.--------------------------------------
!计算功率谱: l=0
dk=(kmax-kmin)/nkw
allocate(pg0(nkw));allocate(NbinKW(nkw))
nbinkw=0     ;     pg0=0.
DO j=1,nx
    !---------------------------
    if(j<=Nx/2)then
        fx = (j-1)/(Nx*dx)
    else
        fx = (j-1-Nx)/(Nx*dx)
    endif
    DO k=1,ny
       if(k<=Ny/2)then
           fy = (k-1)/(Ny*dy)
       else
           fy = (k-1-Ny)/(Ny*dy)
       endif
       DO i=1,nz/2
           fz = (i-1)/(Nz*dz)
           kw=2.*pi*sqrt(fx*fx+fy*fy+fz*fz)
           ikw=(kw-kmin)/dk
    !---------------------------
           IF( (ikw>=0).and.(ikw<nkw).and.(kw<(0.5*min_nqu)) )then
              sincx=1.;sincy=1.;sincz=1.
              if(fx .ne. 0.) sincx = sin(fx*dx*pi)/(fx*dx*pi);
              if(fy .ne. 0.) sincy = sin(fy*dy*pi)/(fy*dy*pi);
              if(fz .ne. 0.) sincz = sin(fz*dz*pi)/(fz*dz*pi);
              grid_cor=1./(sincx*sincy*sincz);
              Pg0(ikw+1) =Pg0(ikw+1)+ (real(FUN_out(i,j,k))**2   &
                     +aimag(FUN_out(i,j,k))**2-PnoiseR)*grid_cor*grid_cor
              if((ikw+1).eq.1)NormR=real(FUN_out(i,j,k))**2&
                                 +aimag(FUN_out(i,j,k))**2-PnoiseR
              nbinkw(ikw+1)=nbinkw(ikw+1)+1
           ENDIF
       ENDDO
   ENDDO
ENDDO





!4.------------------------------------
!归一化:
do ikw=1,Nkw
    if( nbinkw(ikw)>0. ) then
        Pg0(ikw) = Pg0(ikw)/(nbinkw(ikw));
        Pg0(ikw) = Pg0(ikw)/NormR
    endif
enddo
!将最后一个bin设置为零:
nbinkw(nkw)=0.
!输出文件:
open (unit=401123,file=output_dir,form='formatted')
do i=1,nkw
  if(real(Pg0(i))>0.)then
    if(i==nkw)then
      write(401123,*)i, real(kmin+((i-1)+0.5)*dk),0,0
    else
      write(401123,*)i, real(kmin+((i-1)+0.5)*dk),real(Pg0(i)),nbinkw(i)
    endif
  endif
enddo
close(4401123)




!---------------------
deallocate(FUN);deallocate(pg0);deallocate(NbinKW);
end subroutine rand_PS
