% Solve "rigid placement" at station level
function [data, U, GAP]=modelStationRigidPlacement(data, stationData, stationID)

% Inputs:
% data: model
% stationData(stationID): station data
% stationID: station ID (integer)
        
% Outputs:
% data: udpated model model with geometry and inputs
        % refresh existing geometry
        % do not update stiffness matrix
        % do not recompute part UCS
        % only refresh part features, such as hole and slots
% U: deviation field [x, y, z] (nnode, 3)        
% GAP: gap field [1 x nStation]

%
% Step(0) assign placement matrix
nParts=length(data.Input.Part);
for partID=1:nParts
    data.Input.Part(partID).Placement.T=data.Input.Part(partID).Placement.TStore{stationID};
end
%
% Step(1): Link UCS systems between parts
sourceP=stationData(stationID).UCS.Source;
DestinationP=stationData(stationID).UCS.Destination;
if length(sourceP)~=length(DestinationP)
    error('Rigid placement (UCS link): no. of source and destination parts do not match!')
end
np=length(sourceP);
for i=1:np
    if data.Input.Part(DestinationP(i)).Status==0 && data.Input.Part(sourceP(i)).Status==0 &&...
            data.Input.Part(DestinationP(i)).Enable && data.Input.Part(sourceP(i)).Enable 
        
       data.Input.Part(DestinationP(i)).Placement.UCS=data.Input.Part(sourceP(i)).Placement.UCS;
       data.Input.Part(DestinationP(i)).Placement.T=data.Input.Part(sourceP(i)).Placement.T;       
    
    end
end
%
% Step(2): Build reference model and compute outputs
XYZb=data.Model.Nominal.xMesh.Node.Coordinate; % before placement
data=modelBuildPart(data,[0 0 1 1 0]); 
data=modelBuildInput(data,[0 1 1]);
XYZa=data.Model.Nominal.xMesh.Node.Coordinate; % after placement
%
% Save back the deviation field
uvw=XYZa-XYZb;
%
% Compute U
U=zeros(data.Model.Nominal.Sol.nDoF,1);
iddofs=data.Model.Nominal.xMesh.Node.NodeIndex;
U(iddofs(:,1))=uvw(:,1); % X
U(iddofs(:,2))=uvw(:,2); % Y
U(iddofs(:,3))=uvw(:,3); % Z              
%
% Compute GAP
GAP=modelGetGapParts(data);
%
% Step(3): reset UCS
for partID=1:nParts
    data.Input.Part(partID).Placement.UCS=data.Input.Part(partID).Placement.UCSreset;
end
