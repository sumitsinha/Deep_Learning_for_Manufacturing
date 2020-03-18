% run deselect current node selection
function fem=femSetDeselection(fem)

%-----------------------------
% initial values
net=length(fem.xMesh.Element);
nnode=size(fem.xMesh.Node.Coordinate,1);

fem.Selection.Element.Status=false(1,net);
fem.Selection.Node.Status=false(1,nnode);

fem.Selection.Element.Tria.Id=[];
fem.Selection.Element.Tria.Element=[];

fem.Selection.Element.Quad.Id=[];
fem.Selection.Element.Quad.Element=[];

fem.Selection.Element.Tria.Count=[];
fem.Selection.Element.Quad.Count=[];

fem.Selection.Node.Active=[];
fem.Selection.Node.Count=0;
fem.Selection.Node.Boundary=[];
%-----------------------------
