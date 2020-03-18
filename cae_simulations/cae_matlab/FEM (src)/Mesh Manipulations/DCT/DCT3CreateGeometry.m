% rebuild geometry based on voxel deviations
function [fem,data]=DCT3CreateGeometry(fem, data)

n=size(data.NoEmpVoxelId,1);

% set initial value for the node deviations
idpart=fem.Dct.Domain;
nnode=length(fem.Domain(idpart).Node);

data.NodeDev=zeros(nnode,1);

for i=1:n
    
    % id voxel
    idvoxel=data.NoEmpVoxelId(i,:);
    
    % id node
    idnode=data.VoxelId{idvoxel(1),idvoxel(2),idvoxel(3)};
    
    % deviation
    devi=data.CoeffInv(idvoxel(1),idvoxel(2),idvoxel(3));
                
    % update related points
    for t=1:length(idnode) % loop over all points belonging to voxel "idvoxel"

        fem.xMesh.Node.Coordinate(idnode(t),:)=fem.xMesh.Node.Coordinate(idnode(t),:)+devi*fem.xMesh.Node.Normal(idnode(t),:);

        data.NodeDev(idnode(t),1)=devi;

    end
                
end
