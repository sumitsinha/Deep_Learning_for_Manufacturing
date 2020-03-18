% calculate normal vectors for each node (DEFORMED FRAME)
function fem=getNodeNormalDeformedFrame(fem)

nnode=size(fem.Sol.DeformedFrame.Node.Coordinate,1);

for i=1:nnode
    
    Nn=[0 0 0];
    
    idele=fem.Sol.Node2Element{i};
    
    nele=length(idele);
    for j=1:nele
        Nn=Nn+fem.Sol.DeformedFrame.Element(idele(j)).Tmatrix.Normal;   
    end
    
    %--
    Nn=Nn/nele;
    Nn=Nn/norm(Nn);
    
    % save normal
    fem.Sol.DeformedFrame.Node.Normal(i,:)=Nn;
    
    fem.Sol.DeformedFrame.Node.NormalReset(i,:)=Nn;
    
end