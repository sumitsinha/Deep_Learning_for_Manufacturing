% Create sparse matrix
function A=femSparse(irow, icol, Xnz, nTot, nTot)
%
% ARGUMENTs:
%
% Inputs:
% * irow: row index of non zero entries (nx1) - integer
% * icol: row index of non zero entries (nx1) - integer
% * Xnz: vector of non zero entries (nx1) - double
% * nTot: no. of non zero entries - integer
%
% Outputs:
% * A - sparse matrix (nTot x nTot)
%
% compile: part of the suisparse package - http://faculty.cse.tamu.edu/davis/suitesparse.html