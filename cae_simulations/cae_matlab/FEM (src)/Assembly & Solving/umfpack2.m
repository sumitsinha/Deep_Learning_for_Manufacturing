% Solve system of linear using cholesky decomposition.
% Solve x=A-1 * b for a sparse matrix, which do not need to be triangulat or semi-definite positive
function x=umfpack2(A, opt, b)
%
% "umfpack2" is the suggested solver when using the "Lagrange multiplier"
% method to handle constraints in VRM
%
% ARGUMENTs:
%
% Inputs:
% * A: coefficient matrix (n x n) - sparse/double
% * opt = '/'
% * b: load vector - (nx1)/double
%
% Outputs:
% * x - solution vector (nx1)/double
%
% compile: part of the suisparse package - http://faculty.cse.tamu.edu/davis/suitesparse.html