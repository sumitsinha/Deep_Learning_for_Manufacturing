% Mean face normal filter for smoothing/denoising triangular meshes
function fem=meanFilterMesh(fem)
%
% ARGUMENTs:
%
% Inputs:
% fem: fem model with the following fields:
    % .Denoise.Options.MaxIter: max of smoothing iterations (double)
    % .Denoise.Options.Domain: list of domains to be soothed => domain IDs
% The function implements the method develop in "https://ieeexplore.ieee.org/document/1027503"
%
% Outputs:
% fem: udpated fem model with the following fields updated:
    % . xMesh.Node.Coordinate[Only those nodes belonging to "fem.Denoise.Options.Domain"]
%
% compile: mex meanFilterMesh.cpp preProcessingLib.cpp -largeArrayDims 
% Note: use "mex -g" to run in debug mode

    
