function fem=setContactConstraint(fem)
%
% ARGUMENTs:
%
% Inputs:
% * fem: input fem model with definition of following fields
    % fem.Boundary.ContactPair(id).Master % (integer) - master component
    % fem.Boundary.ContactPair(id).MasterFlip % (boolean) - flip master normal vector
    % fem.Boundary.ContactPair(id).Slave (integer) slave component
    % fem.Boundary.ContactPair(id).SearchDist % (double) - searching distance between slave node and master element. Contact pair equation is added if the computed angle is smaller than "SearchDist"
    % fem.Boundary.ContactPair(id).SharpAngle % (double)- angle between normal vector of slave node and normal vector of master element. Contact pair equation is added if the computed angle is smaller than "SharpAngle"
    % fem.Boundary.ContactPair(id).Offset % (double) - offset between master and slave
    % fem.Boundary.ContactPair(id).Enable % (boolean) - If false, no. constraint is added; however the contact pair will be processed in the post-processing for gap calculation
    % fem.Boundary.ContactPair(id).Sampling % (double) [0, 1] - downsampling value of number of contact points 
    % fem.Boundary.ContactPair(id).Frame "ref"; "def".
        % If "def" is used => geometry is selected from "fem.Sol.DeformedFrame"
        % If "ref" is used => geometry is selected from "fem.xMesh"
%
% Outputs:
% * fem: updated fem model with:
    % fem.Boundary.ContactPair(id).Pms (nx3) - coordinate of master contact points
    % fem.Boundary.ContactPair(id).Psl (nx3) - coordinate of master contact points
    % fem.Boundary.ContactPair(id).MasterId (nx1) - IDs of master points
        % If "Type"==1 => MasterId is the node ID    
        % If "Type"==2 => MasterId is the element ID
    % fem.Boundary.ContactPair(id).SlaveId (nx1) - IDs of slave points
    % fem.Boundary.ContactPair(id).Type (nx1) - 0/1/2 => not assigned / node-to-node / node-to-element
%
% compile: mex setContactConstraint.cpp preProcessingLib.cpp shapeFcn.cpp -largeArrayDims COMPFLAGS="/openmp $COMPFLAGS"
% Note: use "mex -g" to run in debug mode

    
