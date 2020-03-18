% Set geometry to deformed
function data=modelSetGeometryToDeformed(data)
%
% Inputs
% data: data model
%
% Outputs
% data: updated data model with:
        % deformed geometry
        % part features
%
%--

%-----
IDsol=1; % Only take the first solution set
%-----

if ~isfield(data.Input,'Part')
    return
end
%
nParts=length(data.Input.Part);
for i=1:nParts
    if data.Input.Part(i).Enable && data.Input.Part(i).Status==0 % active and computed
         % displacement field
         uvw=data.Input.Part(i).U{IDsol}(:,1:3);
         % Node ids
         nodeIDi=data.Model.Nominal.Domain(i).Node;
         nodei=data.Model.Nominal.xMesh.Node.Coordinate(nodeIDi,:);
         % udpate
         data.Model.Nominal.xMesh.Node.Coordinate(nodeIDi,:)=nodei+uvw;
     end
end
%
% Refresh all
data=modelBuildPart(data,[0 0 0 0 0]); 
%
% Update part features
field={'Stitch'};
for i=1:length(field)
    data=updateLocal(data, field{i}, IDsol);
end
%
% Refresh all
data=modelBuildInput(data,[0 1 0]);

%------------------------------------
%--
function data=updateLocal(data, field, IDsol)

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
        fem.Sol.U=data.Assembly.U{IDsol};
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
       
