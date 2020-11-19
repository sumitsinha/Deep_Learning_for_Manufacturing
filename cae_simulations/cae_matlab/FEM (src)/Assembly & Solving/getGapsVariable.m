% Compute gaps from contact pairs
function fem=getGapsVariable(fem)
%
% ARGUMENTs:
%
% Inputs:
% * fem - input fem model with the following fields pre-computed/assigned:
    % fem.Boundary.ContactPair(id).Master % (integer) - master component
    % fem.Boundary.ContactPair(id).MasterFlip % (boolean) - flip master normal vector
    % fem.Boundary.ContactPair(id).Slave (integer) slave component
    % fem.Boundary.ContactPair(id).SearchDist % (double) - searching distance between slave node and master element. Contact pair equation is added is the computed angle is smaller than "SearchDist"
    % fem.Boundary.ContactPair(id).SharpAngle % (double)- angle between normal vector of slave node and normal vector of master element. Contact pair equation is added if the computed angle is smaller than "SharpAngle"
    % fem.Boundary.ContactPair(id).Offset % (double) - offset between master and slave
    % fem.Boundary.ContactPair(id).Enable % (boolean) - If false, no. constraint is added; however the contact pair will be processed in the post-processing for gap calculation
    % fem.Boundary.ContactPair(id).Sampling % (double) [0, 1] - downsampling value of number of contact points 
    % fem.Options.GapFrame "ref"; "def".
        % If "def" is used => gap is computed by addding to the geometry distance the deformation stored in "fem.Sol.U"
        % If "ref" is used => only use the geometric distance/reference uing geometry stored in "fem.xMesh"
    % fem.Sol.U => solution vector
%
% Outputs:
% * fem - output fem model with the following fields computed:
    % .Sol.Gap(id).Gap - computed gaps values for the given contact pair "id"
        % The structure needs to be initialised for each contact pair, "id"
           % .Sol.Gap(id).Gap=zeros(1,nnode) % gap for each contact pair (nnode= total no. of nodes)
           % .Sol.Gap(id).max=fem.Options.Min
           % .Sol.Gap(id).min=fem.Options.Max
%
% compile: mex getGapsVariable.cpp preProcessingLib.cpp shapeFcn.cpp -largeArrayDims COMPFLAGS="/openmp $COMPFLAGS"
% Note: use "mex -g" to run in debug mode
