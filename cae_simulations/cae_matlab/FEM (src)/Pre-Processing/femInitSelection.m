% initialise selection
function fem=femInitSelection(fem)

% initial values
nele=length(fem.xMesh.Element);
nnode=size(fem.xMesh.Node.Coordinate,1);

fem.Selection.Element.Status=false(1,nele);
fem.Selection.Node.Status=false(1,nnode);

% loop over all domains
ntria=0;
nquad=0;

fem.Selection.Element.Tria.Id=[];
fem.Selection.Element.Tria.Element=[];

fem.Selection.Element.Quad.Id=[];
fem.Selection.Element.Quad.Element=[];

idnode=[];
nd=fem.Sol.nDom;
for i=1:nd
    if fem.Domain(i).Status
        
        % node--
        idnode=[idnode, fem.Domain(i).Node];
        
        % element--
        fem.Selection.Element.Status(fem.Domain(i).Element)=true;
        
        % save trias
        fem.Selection.Element.Tria.Id=[fem.Selection.Element.Tria.Id,...
                                       fem.Domain(i).ElementTria];
                                   
        elei=[fem.xMesh.Element(fem.Domain(i).ElementTria).Element];
        fem.Selection.Element.Tria.Element=[fem.Selection.Element.Tria.Element;...
                                            reshape(elei,3,length(fem.Domain(i).ElementTria))'];
                                        
        ntria=ntria+length(fem.Domain(i).ElementTria);
        
        % save quads
        fem.Selection.Element.Quad.Id=[fem.Selection.Element.Quad.Id,...
                                       fem.Domain(i).ElementQuad];
                                   
        elei=[fem.xMesh.Element(fem.Domain(i).ElementQuad).Element];
        fem.Selection.Element.Quad.Element=[fem.Selection.Element.Quad.Element;...
                                            reshape(elei,4,length(fem.Domain(i).ElementQuad))'];
                                        
        nquad=nquad+length(fem.Domain(i).ElementQuad);

    end 
end

% save out
fem.Selection.Element.Tria.Count=ntria;
fem.Selection.Element.Quad.Count=nquad;

%--
fem.Selection.Node.Status(idnode)=true;
fem.Selection.Node.Active=idnode;
fem.Selection.Node.Count=length(idnode);
fem.Selection.Node.Boundary=[];

