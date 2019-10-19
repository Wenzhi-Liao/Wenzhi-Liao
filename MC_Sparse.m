function result= MC_Sparse(spectraldata,lambda,group,iterations)
% result= MC_Sparse(spectraldata,lambda,group,iterations,op)
%       argmin  lambda*||Kx-y||2 +||TV(x)||p,   where p=1;
% perform denoising and deblurring on hyperspectral images based on multi-component
% sparsity
% 
%    Input:
%          spectraldata   - original (noisy) hyperspectral images, with r
%                           samples, c lines and b spectral bands
%
%          lambda         - parameter to control fidelity function
%
%          group          - parameter to control sparsity transform for
%                           each band seperate (group=0), or structured 
%                           sparsity (group=1)
%
%          iterations     - number of iteration 
%
%
%
%     Output:
%            result       - the denoised and deblurred hyperspectral image
%
%
%     one example:
%                 load IT1ms_downSample6; 
%                 spectraldata=double(spectraldata);
%                 lambda=0.01; group=1; iterations=15;
%                 spectraldataclean =MC_Sparse(spectraldata,lambda,group,iterations);  
%
%
%
% Copyright 2012-2016
% UGent-Telin-iMinds-IPI
% Contact: wenzhi.liao@telin.ugent.be 
% http://telin.ugent.be/~wliao/
% 2 Feb 2012
%

[r,c,b] = size(spectraldata);

%%%% initialization %%%%%%%%%%
result_primal = spectraldata; 
result_dualx = Dx(result_primal);
result_dualy = Dy(result_primal);
result_predict = result_primal;

t = 20; % dual stepsize 
s = 1/(t*8); % s*t*L2(grad) <= 1   met L2(grad)=8
theta = 0.05; % voorspelling



for iter = 1:iterations 
    
    expr_dualx = result_dualx + s*Dx(result_predict);
    expr_dualy = result_dualy + s*Dy(result_predict);
    if (group == 0)
        for i = 1:b
            result_dualx(:,:,i) = expr_dualx(:,:,i)./(max(1,abs(expr_dualx(:,:,i)))); % monotone operator
            result_dualy(:,:,i) = expr_dualy(:,:,i)./(max(1,abs(expr_dualy(:,:,i)))); % monotone operator
        end
    else
        normx = 0;
        normy = 0;
        for i = 1:b
            normx = normx + expr_dualx(:,:,i).^2;
            normy = normy + expr_dualy(:,:,i).^2;
        end
        normx = sqrt(normx);
        normy = sqrt(normy);
        for i = 1:b
            result_dualx(:,:,i) = expr_dualx(:,:,i)./(max(1,normx)); % monotone operator
            result_dualy(:,:,i) = expr_dualy(:,:,i)./(max(1,normy)); % monotone operator
        end
    end
    result_primal2 = result_primal;
    expr_primal = result_primal - t*Dxt(result_dualx) - t*Dyt(result_dualy);
    result_primal = (expr_primal + t*lambda*spectraldata)./(1 + t*lambda); % monotone operator
    result_predict = result_primal + theta*(result_primal - result_primal2);
end

result = result_primal;

return

function d = Dx(u)
% d = Dx2(u)
% calculate the gradient of image u on x direction
%
[rows,cols,color] = size(u); 
d = zeros(rows,cols,color);
d(:,2:cols,:) = u(:,2:cols,:)-u(:,1:cols-1,:);
d(:,1,:) = u(:,1,:)-u(:,cols,:);

return

function d = Dxt(u)
% d = Dxt2(u)
% calculate the divergence of image u on x direction
%
[rows,cols,color] = size(u); 
d = zeros(rows,cols,color);
d(:,1:cols-1,:) = u(:,1:cols-1,:)-u(:,2:cols,:);
d(:,cols,:) = u(:,cols,:)-u(:,1,:);

return

function d = Dy(u)
% d = Dy2(u)
% calculate the gradient of image u on y direction
[rows,cols,color] = size(u); 
d = zeros(rows,cols,color);
d(2:rows,:,:) = u(2:rows,:,:)-u(1:rows-1,:,:);
d(1,:,:) = u(1,:,:)-u(rows,:,:);

return

function d = Dyt(u)
% d = Dxt2(u)
% calculate the divergence of image u on y direction
%
[rows,cols,color] = size(u); 
d = zeros(rows,cols,color);
d(1:rows-1,:,:) = u(1:rows-1,:,:)-u(2:rows,:,:);
d(rows,:,:) = u(rows,:,:)-u(1,:,:);

return
