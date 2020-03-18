% Set domain load conditions. It solve the integral:
    % F=integral(q), where
        % F: nodal forces
        % q: distributed load on the element/domain level
function fem=setDomainLoad(fem)
%
% ARGUMENTs:
%
% Inputs:
% * fem: input fem model with the following fields
    % .Domain(id).Load.Value: [1,6] =>[Fx, Fy, Fz, Mx, My, Mz]
    % .Domain(id).Load.Flag: True/False => compute/do not compute
%
% Outputs:
% * fem: updated fem model with:
    % .Boundary.Load.DofId
    % .Boundary.Load.Value                   
%
% compile: mex setDomainLoad.cpp preProcessingLib.cpp shapeFcn.cpp -largeArrayDims 
% Note: use "mex -g" to run in debug mode

