% Generate non-ideal parts at station level
function [data, U, GAP]=modelStationNonidealParts(data, stationData, stationID)

% Inputs:
% data: model
% stationData(stationID): station data
% stationID: station ID (integer)
        
% Outputs:
% data: udpated model
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
                 if geom==2 % morphing
                    [uvwD, ~]=morphGeometrySolve(data, i);
                    data.Input.Part(i).Geometry.Parameter=1;
                    data.Input.Part(i).D{1}=uvwD;
                 elseif geom==3 % measured
                    ppart=data.Input.Part(i).Geometry.Parameter;
                    uvwD=data.Input.Part(i).D{ppart};
                 end
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
data=modelSetGeometryToDeformed(data, stationData, stationID, 2, U);
%
% Compute GAP
GAP=modelGetGapParts(data);
