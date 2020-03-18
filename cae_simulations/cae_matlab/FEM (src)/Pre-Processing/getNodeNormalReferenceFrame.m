% calculate normal vectors for each node (REFERENCE FRAME)
function fem=getNodeNormalReferenceFrame(fem)

nnode=size(fem.xMesh.Node.Coordinate,1);

for i=1:nnode
    
    Nn=[0 0 0];
    
    idele=fem.Sol.Node2Element{i};
    
    nele=length(idele);
    for j=1:nele
          Nn=Nn+fem.xMesh.Element(idele(j)).Tmatrix.Normal;    
    end
    
    % get normal vector by average method
    Nn=Nn/nele;
    Nn=Nn/norm(Nn);
    
    % save normal
    fem.xMesh.Node.Normal(i,:)=Nn;
    
    fem.xMesh.Node.NormalReset(i,:)=Nn;
    
end