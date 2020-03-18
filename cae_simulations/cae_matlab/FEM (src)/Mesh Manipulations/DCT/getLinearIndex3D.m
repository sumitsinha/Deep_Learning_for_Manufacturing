% get linear index
function t=getLinearIndex3D(i, j, k, n, m)

% n/m= no. od row/columns
%-----------------------------------------------
% use column-wide ordering (stack-up of columns)
%-----------------------------------------------
t=(m*n)*(k-1) +  n*(j-1)+i;