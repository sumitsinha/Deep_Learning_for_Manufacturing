% Set geometry to deformed
function data=modelSetGeometryToDeformed(data, stationData, stationID, opt, U)
%
% Inputs
% data: data model
% stationData(stationID): station data
% stationID: station ID (integer)
% opt:
    % 1: use U field
    % 2: use D field
% Deviation field (either U or D field)

%
% Outputs
% data: updated data model with:
        % deformed geometry
        % part features
%
%--
if nargin==3
    opt=1;
    U=[];
end

%-----
IDsol=1; % Only take the first solution set
%-----

if ~isfield(data.Input,'Part')
    return
end
%
for i=stationData(stationID).Part
    if data.Input.Part(i).Enable && data.Input.Part(i).Status==0 % active and computed
         % displacement field
         if opt==1
            uvw=data.Input.Part(i).U{IDsol}(:,1:3);
         elseif opt==2
            ppart=data.Input.Part(i).Geometry.Parameter;
            uvw=data.Input.Part(i).D{ppart};
         end
         if ~isempty(uvw)
             % Node ids
             nodeIDi=data.Model.Nominal.Domain(i).Node;
             nodei=data.Model.Nominal.xMesh.Node.Coordinate(nodeIDi,:);
             % udpate
             data.Model.Nominal.xMesh.Node.Coordinate(nodeIDi,:)=nodei+uvw;
         end
     end
end
%
% Refresh all
data=modelBuildPart(data,[0 0 0 0 0]); 
%
% Update part features
if~isempty(U)
    field={'Stitch'};
    for i=1:length(field)
        data=updateLocal(data, field{i}, U);
    end
end
%
% Refresh all
data=modelBuildInput(data,[0 1 0]);
%
%------------------------------------
%--
function data=updateLocal(data, field, U)

%---------------------
fem=data.Model.Nominal;
%---------------------

f=getInputFieldModel(data, field);
n=length(f);

for i=1:n
    fi=f(i);  
    
    if fi.Enable
        Pmi=fi.Pm;
        %
        fem.Sol.U=U;
        fem.Post.Interp.Pm=Pmi; 
        fem.Post.Interp.Domain=fi.DomainM;
        fem.Post.Interp.SearchDist=fi.SearchDist(1);

        ufield=zeros(size(Pmi,1),3);
        intVar={'u','v','w'};
        for j=1:3
            fem.Post.Interp.InterpVariable=intVar{j};
            fem.Post.Interp.ContactPair=0;

            [~, ufield(:,j), ~]=getInterpolationData_fast(fem);
        end
        % Update Pm
        fi.Pm=Pmi + ufield;

        % save back
        data=retrieveBackStructure(data, fi, field, i);
    end
end
       
