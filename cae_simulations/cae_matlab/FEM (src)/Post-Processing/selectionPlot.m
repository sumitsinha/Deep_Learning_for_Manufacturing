% plot current selection
function selectionPlot(fem)

% PLOT TRIA:
ntria=fem.Selection.Element.Tria.Count;

if ntria>0
    
    patch('faces',fem.Selection.Element.Tria.Element,...
     'vertices',fem.xMesh.Node.Coordinate,...
     'edgecolor','k',...
     'facecolor','r',...
     'parent',fem.Post.Options.ParentAxes,...
     'tag','tempobj')    
 
end


% PLOT QUAD:
nquad=fem.Selection.Element.Quad.Count;

if nquad>0
    
    patch('faces',fem.Selection.Element.Quad.Element,...
     'vertices',fem.xMesh.Node.Coordinate,...
     'edgecolor','k',...
     'facecolor','r',...
     'parent',fem.Post.Options.ParentAxes,...
     'tag','tempobj') 
 
end

% STEP 2: plot boundary
bnode=fem.Selection.Node.Boundary;
bpoint=fem.xMesh.Node.Coordinate(bnode,:);

% plot
if ~isempty(bpoint)
    
    line('xdata',bpoint(:,1),...
      'ydata',bpoint(:,2),...
      'zdata',bpoint(:,3),...
      'marker','o',...
      'markersize',6,...
      'markerfacecolor','c',...
      'linestyle','none',...
      'parent',fem.Post.Options.ParentAxes,...
      'tag','tempobj')
    
end


% %
% view(3)
% axis equal

if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end

        