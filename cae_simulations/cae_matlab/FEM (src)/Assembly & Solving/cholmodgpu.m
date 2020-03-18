% Solve system of linear using cholesky decomposition.
% Code optimised for multi-core GPU computation

% x=A-1 * b
function x=cholmodgpu(A, b)
%
% "cholmod" is the suggested solver when using the "Penaltry method"
% method to handle constraints in VRM
% 
% ARGUMENTs:
%
% Inputs:
% * A: coefficient matrix (n x n) - sparse/double
    % A is symmetric and positive definite
% * b: load vector - (nx1)/double
%
% Outputs:
% * x- solution vector (nx1)/double
%
% compile: part of the suisparse package - http://faculty.cse.tamu.edu/davis/suitesparse.html