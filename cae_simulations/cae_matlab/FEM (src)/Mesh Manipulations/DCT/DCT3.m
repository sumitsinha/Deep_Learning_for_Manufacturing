% calculate dct coefficient for 3D data voxel
function B = DCT3(A)    

% data size
[M, N, T] = size(A);
B = zeros(M,N,T);

% get the coefficient
dctmtx_N = dctmtx(N);
dctmtx_M = dctmtx(M);
dctmtx_T = dctmtx(T);

% update
for t=0:T-1
    B(:,:,t+1) = dctmtx_M * A(:,:,t+1) * dctmtx_N';
end

% reshape and save all
B = reshape(B,[],T);
B = B * dctmtx_T';
B = reshape(B,M,N,T);