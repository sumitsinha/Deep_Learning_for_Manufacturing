% offset mesh model
function fem=femMeshOffset(fem, idpart, offvalue)

% define options
femc.Options.StiffnessUpdate=false;
femc.Options.MassUpdate=false;
femc.Options.ConnectivityUpdate=false;

ntot=size(fem.xMesh.Node.Coordinate,1);
dev=zeros(ntot,1);

% calculate deviations
count=1;
for i=idpart
    id=fem.Domain(i).Node;
    dev(id)=offvalue(count);
    count=count+1;
end

% create new mesh
fem=createVariationalMesh(fem, dev);

% update
fem=femPreProcessing(fem);