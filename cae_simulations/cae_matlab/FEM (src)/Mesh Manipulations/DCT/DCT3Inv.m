% calculate inverse dct on 3D data voxel
function iB = DCT3Inv(B)    

%
[P, Q, R] = size(B);

iB = zeros(P,Q,R);

% get coefficient
dctmtx_P = dctmtx(P);
dctmtx_Q = dctmtx(Q);
dctmtx_R = dctmtx(R);

for t=0:R-1
    iB(:,:,t+1) = dctmtx_P' * B(:,:,t+1) * dctmtx_Q;
end

iB = reshape(iB,[],R);
iB = iB * dctmtx_R;
iB = reshape(iB,P,Q,R);   
   