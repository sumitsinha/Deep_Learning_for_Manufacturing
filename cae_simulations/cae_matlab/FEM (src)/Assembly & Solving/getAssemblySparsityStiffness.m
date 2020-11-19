% Compute assembly sparsity of stiffness matrix
function [irow, icol, Xnz, Fmod]=getAssemblySparsityStiffness(fem)
%
% It computes the assembly sparsity with conditioning (if any constraint available). The
% assembly sparsity has the following properties:
    % if using the penalty method => symmetric and semi-positive definite
    % is using the lagrange multiplier method => only symmetric
%
% ARGUMENTs:
%
% Inputs:
% * fem - input fem model with the following fields pre-computed:
    % .Sol
    % .Boundary.Constraint.ContactWset => unilateral pairs
    % .Boundary.Constraint.MPC => multi-point constraints
    % .Boundary.Constraint.SPC => single-point constraints
    % .Boundary.Load => load conditions
    % .xMesh.Element.Ke => stiffness matrix of individual element
        % make sure "fem.Options.StiffnessUpdate" is true when calling "femPreProcessing"
%
% Outputs:
% * irow: row index of non zero entries (nx1) - integer
% * icol: row index of non zero entries (nx1) - integer
% * Xnz: vector of non zero entries (nx1) - double
% * Fmod: modified load vector - double
%
% compile: getAssemblySparsityStiffness.cpp -largeArrayDims
% Note: use "mex -g" to run in debug mode
