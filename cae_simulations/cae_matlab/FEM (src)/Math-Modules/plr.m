function R = plr(A)
%PLR Compute polar decomposition
%
% R = plr(A) return the closet rotation matrix to A in Frobenius norm

[U,~,V] = svd(A);

R = U * diag([1,1,det(U*V')]) * V';