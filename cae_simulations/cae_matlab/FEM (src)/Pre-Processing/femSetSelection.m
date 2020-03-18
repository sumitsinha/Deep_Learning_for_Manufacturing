% set selection based on "idnode"
function fem=femSetSelection(fem, activeNode)

% activeNode(id).Node: list of nodes to be activated
% activeNode(id).Status: -1/0 NOT active/active
% activeNode(id).Part: part ID

idnode=selection2Nodes(fem, activeNode);

% ... then select
if ~isempty(idnode)
    fem=setSelectedNode(fem, idnode);
else % deselect all
    fem=femSetDeselection(fem);
end   

% set selected nodes
function fem=setSelectedNode(fem, idnode)

% initial values
net=length(fem.xMesh.Element);
nnode=size(fem.xMesh.Node.Coordinate,1);

fem.Selection.Element.Status=false(1,net);
fem.Selection.Node.Status=false(1,nnode);

%--
% elements
idele=unique(cell2mat(fem.Sol.Node2Element(idnode)));
fem.Selection.Element.Status(idele)=true;

% count elements
[nquadt, ntriat]=countTriaQuad(fem, idele);

nele=length(idele);

ntria=0;
nquad=0;

fem.Selection.Element.Tria.Id=zeros(1,ntriat);
fem.Selection.Element.Tria.Element=zeros(ntriat,3);

fem.Selection.Element.Quad.Id=zeros(1,nquadt);
fem.Selection.Element.Quad.Element=zeros(nquadt,4);
for i=1:nele
    
    etype=fem.xMesh.Element(idele(i)).Type;
    elei=fem.xMesh.Element(idele(i)).Element;
    
    if strcmp(etype,'tria')
        ntria=ntria+1;
        
        fem.Selection.Element.Tria.Id(ntria)=i;
        fem.Selection.Element.Tria.Element(ntria,:)=elei;
    elseif strcmp(etype,'quad')
        nquad=nquad+1;
        
        fem.Selection.Element.Quad.Id(nquad)=i;
        fem.Selection.Element.Quad.Element(nquad,:)=elei;
    end
    
end

fem.Selection.Element.Tria.Count=ntriat;
fem.Selection.Element.Quad.Count=nquadt;

%--

%--
% nodes
activenode=unique([fem.xMesh.Element(idele).Element]);

fem.Selection.Node.Status(activenode)=true;
fem.Selection.Node.Active=activenode;
fem.Selection.Node.Count=length(activenode);

% get boundaries
bnode=getBoundaryNode(fem);
fem.Selection.Node.Boundary=bnode;

%--
function [nquad, ntria]=countTriaQuad(fem, idele)

nele=length(idele);

ntria=0;
nquad=0;

for i=1:nele
    
    etype=fem.xMesh.Element(idele(i)).Type;
    
    if strcmp(etype,'tria')
        ntria=ntria+1;
    elseif strcmp(etype,'quad')
        nquad=nquad+1;
    end
    
end

%--
function idnode=selection2Nodes(fem, activeNode)

idnode=[];

if ~isempty(activeNode) 
   
    % loop over all domains
    nd=length(activeNode);

    for i=1:nd
       if activeNode(i).Status % active domain
           if ~isempty(activeNode(i).Node) % use selected nodes
               idnode=[idnode, activeNode(i).Node];
           end
       end
    end
    
end

 % use all node
if isempty(activeNode) || isempty(idnode)
    
    nnode=size(fem.xMesh.Node.Coordinate,1);
    idnode=1:nnode;
   
end


