% Solve system of linear equations using cholesky decomposition.
% It used the pardiso direct solver

% x=A-1 * b
function x=call_pardiso_symm_pos_def(A, b)
%
% ARGUMENTs:
%
% Inputs:
% * A: coefficient matrix (n x n) - sparse/double
    % A is symmetric and positive definite and upper triangular
% * b: load vector - (nx1)/double
%
% Outputs:
% * x- solution vector (nx1)/double
%
% compile: part of the pardiso distribution - http://pardiso-project.org/manual/pardiso-matlab.tgz
% License file must be requested by the user (not included in the distribution)