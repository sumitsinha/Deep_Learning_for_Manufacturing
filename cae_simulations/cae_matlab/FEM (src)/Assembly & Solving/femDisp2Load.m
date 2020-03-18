% Re-built nodal forces based on nodal displacements
% N.B.: Initial stress due to given node displacements

% sigma=D*B*u
% force=-int(Bt*D*B*u) => force=-K*u

function fem=femDisp2Load(fem, U)
%
% ARGUMENTs:
%
% Inputs:
% * fem: input fem model
% * U: solution vector
%
% Outputs:
% * fem: updated fem model with following field udpated
    % .Boundary.Load.DofId
    % .Boundary.Load.Value
%
% compile: mex femDisp2Load.cpp preProcessingLib.cpp 
% Note: use "mex -g" to run in debug mode

