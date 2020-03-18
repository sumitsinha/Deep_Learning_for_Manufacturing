% get PCA decomposition
function [V, lamda, mu]=getPCADecomposition(X,...
                                            thold)

% X=[ndim, nsample]
% thold: percentage of variance to be removed (<=thold is removed)

% V=[ndim, ndim]... eigenvectors
% lamda=[1, ndim]... eigenvalues
% mu=[1,ndim]... mean vector

% 
[ndim, nsample]=size(X);

% STEP 1: get mean
mu=mean(X,2);

% STEP 2: build Z=1/sqrt(nsample-1) * X - mu;
Z=1/sqrt(nsample-1) * ( X - repmat(mu,1,nsample) );

% STEP 3: get eigenvectors and eigenvalues (USE SVD DECOMPOSITION)
if ndim>=nsample
    [V,s,~] = svd(Z,0); % use "eco" decomposition... only the first "nsample" vectors are collected
else
    [V,s,~] = svd(Z);
end

% get eigenvalues (square of singular value). These correspond to variances
lamda=diag(s).^2;

% STEP 4: get main sources of variance
lamdacum=lamda./sum(lamda);

id=find(lamdacum>=thold);
lamda=lamda(id);
V=V(:,id);



