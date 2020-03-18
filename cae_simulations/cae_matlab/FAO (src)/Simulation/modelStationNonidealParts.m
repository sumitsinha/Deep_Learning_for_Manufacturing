% Generate non-ideal parts at station level
function [data, U, GAP]=modelStationNonidealParts(data, stationData, stationID)

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

% Compute U
U=zeros(data.Model.Nominal.Sol.nDoF,1);
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
                U(iddofs(:,1))=uvwD(:,1); % X
                U(iddofs(:,2))=uvwD(:,2); % Y
                U(iddofs(:,3))=uvwD(:,3); % Z
            end
        end
    end
end
%
% Compute GAP
GAP=modelGetGapParts(data);
