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
% Add contribution of non ideal part
for i=stationData(stationID).Part
    if data.Input.Part(i).Enable && data.Input.Part(i).Status==0
        geom=data.Input.Part(i).Geometry.Type{1};

        if geom>1 % NOT IDEAL GEOMETRY
            ppart=data.Input.Part(i).Geometry.Parameter;
            uvwD=data.Input.Part(i).D{ppart};
            if ~isempty(uvwD)
                idnode=data.Model.Nominal.Domain(i).Node;
                iddofs=data.Model.Nominal.xMesh.Node.NodeIndex(idnode,:);

                % save back
                U(iddofs(:,1))=U(iddofs(:,1)) + uvwD(:,1); % X
                U(iddofs(:,2))=U(iddofs(:,2)) + uvwD(:,2); % Y
                U(iddofs(:,3))=U(iddofs(:,3)) + uvwD(:,3); % Z
            end
        end
    end
end
%
% Compute GAP
GAP=modelGetGapParts(data);
%
% Step(3): reset UCS
np=length(data.Input.Part);
for i=1:np
    data.Input.Part(i).Placement.UCS=data.Input.Part(i).Placement.UCSreset;
end
