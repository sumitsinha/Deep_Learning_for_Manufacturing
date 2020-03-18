% Set domain properties at element level
function fem=setDomainProperties(fem)
%
% ARGUMENTs:
%
% Inputs:
% * fem: input fem model with the following fields
    % .Domain(id).Constant.Th
    % .Domain(id).Material
        % E
        % ni
        % lamda
        % Density
        %....
%
% Outputs:
% * fem: updated fem model with:
    % fem.xMesh.Element(ide).Constant.Th   
    % fem.xMesh.Element(ide).Material.
        % E
        % ni
        % lamda
        % Density
        %....
%
% compile: mex setDomainProperties.cpp
% Note: use "mex -g" to run in debug mode

