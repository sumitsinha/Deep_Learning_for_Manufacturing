% save NODEs
function fem=storeNodes(fem, node, idnodes)

% save NODE
nnode=size(node,1);

% nominal mesh
fem.xMesh.Node.Coordinate=[fem.xMesh.Node.Coordinate,
                           node];
fem.xMesh.Node.CoordinateReset=[fem.xMesh.Node.CoordinateReset,
                           node];

fem.xMesh.Node.Component=[fem.xMesh.Node.Component idnodes];

fem.xMesh.Node.Normal=[fem.xMesh.Node.Normal
                       zeros(nnode,2),ones(nnode,1)];
                   
fem.xMesh.Node.NormalReset=[fem.xMesh.Node.NormalReset
                           zeros(nnode,2),ones(nnode,1)];
                       
fem.xMesh.Node.NodeIndex=[fem.xMesh.Node.NodeIndex
                          zeros(nnode,6)];

% deformed frame
fem.Sol.DeformedFrame.Node.Coordinate=[fem.Sol.DeformedFrame.Node.Coordinate
                                       node];
                                   
fem.Sol.DeformedFrame.Node.Normal=[fem.Sol.DeformedFrame.Node.Normal
                                   zeros(nnode,2),ones(nnode,1)]; 
                               
fem.Sol.DeformedFrame.Node.NormalReset=[fem.Sol.DeformedFrame.Node.NormalReset
                                        zeros(nnode,2),ones(nnode,1)]; 

count=length(fem.xMesh.Node.Tnode)+1;
for i=1:nnode
    fem.xMesh.Node.Tnode{count}=eye(3,3);
    count=count+1;
end