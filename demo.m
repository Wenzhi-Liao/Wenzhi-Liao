%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             
%           Hyperspectral images denoising and deblurring based on multi-component
%           sparsity (MC-Sparse)
%                             
%           
%       Copyright 2012-2016
%       Telin-iMinds-IPI, Ghent University, Belgium
%       Contact: wenzhi.liao@telin.ugent.be
%       Date: 20/07/2012
%       http://telin.ugent.be/~wliao/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;


% Load your data/imread your data
load cube150x170
spectraldata=double(cube(:,:,101:end));
[nrows,ncols, bands]=size(spectraldata);
clear cube
%%%%%%%%%%%%%%%%%%%%% PCA Transformation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rand('state',4711007);% initialization of rand
No_Train=5000; % the number of training samples for PCA
nDim=bands; % the number of PCs
idxtrain = randsample(nrows*ncols, No_Train);
Xo=reshape(spectraldata,nrows*ncols, bands);
X1 = double(Xo(idxtrain,:));
W   = pca(X1, nDim); %eigenvector
FeaPCA=Xo*W; 
clear Xo
outPCA=reshape(FeaPCA,nrows,ncols,nDim);% PCs
clear FeaPCA
%%%%%%% Proposed image restoration in the PCA Domain %%%%%%%%%%%%%
% Parameters settings 
lambda=0.5; % parameter to control fidelity function 
group=1; %parameter to control sparsity transform structured sparsity (group=1)
iterations=15; %number of iterations
tic
noPCs=7; %number of PCs you select
% resoration on the first few PCs using TV with group sparsity
out1=MC_Sparse(outPCA(:,:,1:noPCs),lambda,group,iterations);
% only noise reduction on the rest PCs using solf-threshold denoising
out2=zeros(nrows,ncols,bands-noPCs);
for i=1:bands-noPCs
    [thr,sorh,keepapp] = ddencmp('den','wv',outPCA(:,:,i+noPCs));
    out2(:,:,i) =  wdencmp('gbl',outPCA(:,:,i+noPCs),'sym4',4,thr,sorh,keepapp);%Wavelet solf-threshold denoising
end
spectraldataclean1=cat(3,out1,out2);
clear out1 out2 % release some memory
Y=reshape(spectraldataclean1,nrows*ncols,bands)*W';
clear spectraldataclean1 % release some memory
spectraldataclean=reshape(Y,nrows,ncols,bands); % restorated data
clear Y outPCA
toc

%%%%%%%%%%%%%%%  Display spectral reflection of some cubes %%%%%%%%%%%%%%%
original_pixel=reshape(spectraldata(41,97,:),1,bands); % one cube from hair
clean_pixel=reshape(spectraldataclean(41,97,:),1,bands); % one cube from the pepper seed

i=1:bands;
figure;
plot(i,original_pixel,'-k',i,clean_pixel,'-r');grid on;
xlabel('Spectral Channel')
ylabel('Spectral reflection')
legend('Orig.','Restored')


%%%%%%%%%%%%%%%  Display some denoised bands%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; 
subplot(2,2,1); imshow(spectraldata(:,:,1),[]);
title('Original band 1');
subplot(2,2,2); imshow(spectraldata(:,:,bands),[]);
title('Original last band');
subplot(2,2,3); imshow(spectraldataclean(:,:,1),[min(min(spectraldata(:,:,1))) max(max(spectraldata(:,:,1)))]);
title('Restored band 1');
subplot(2,2,4); imshow(spectraldataclean(:,:,bands),[min(min(spectraldata(:,:,bands))) max(max(spectraldata(:,:,bands)))]);
title('Restored last band');

