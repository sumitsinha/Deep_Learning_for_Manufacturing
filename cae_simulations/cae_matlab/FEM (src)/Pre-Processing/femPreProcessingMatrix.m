function fem=femPreProcessingMatrix(fem)
%
% ARGUMENTs:
%
% Inputs:
% * fem: input fem model
%
% Outputs:
% * fem: updated fem model with:
    % 1: dofs indexes
    % 2: rotation matrices/normal vectors
    % 3: UCS matrices
    % 4: stiffness/mass matrices
    % 5: no. of dofs
%
% compile: mex femPreProcessingMatrix.cpp stiffnessLib.cpp shapeFcn.cpp materialLib.cpp  preProcessingLib.cpp -largeArrayDims  COMPFLAGS="/openmp $COMPFLAGS"
% Note: use "mex -g" to run in debug mode

    
