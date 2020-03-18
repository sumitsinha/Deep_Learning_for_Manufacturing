% plot mesh model
function varargout=meshPlot(fem)

model=initModel2Vrml();
    
% PLOT TRIA:
ntria=fem.Sol.Tria.Count;

if ntria>0

    idnode=unique(fem.Sol.Tria.Element(:));   
    node=fem.xMesh.Node.Coordinate(idnode,:);

    tria=renumberElements(fem.Sol.Tria.Element, idnode);
    
    model.Tria.Face=tria;
    model.Tria.Node=node;
    model.Tria.Trasparency=fem.Post.Options.FaceAlpha;
    model.Tria.Color=fem.Post.Options.ColorPatch;
    model.Tria.Shade='uniform';
    
    if fem.Post.Options.ShowPatch
        if fem.Post.Options.ShowEdge
            patch('faces',tria,...
                 'vertices',node,...
                 'edgecolor',fem.Post.Options.ColorEdge,...
                 'facecolor',fem.Post.Options.ColorPatch,...
                 'parent',fem.Post.Options.ParentAxes,...
                 'facealpha',fem.Post.Options.FaceAlpha)
        else
            patch('faces',tria,...
                 'vertices',node,...
                 'edgecolor','none',...
                 'facecolor',fem.Post.Options.ColorPatch,...
                 'parent',fem.Post.Options.ParentAxes,...
                 'facealpha',fem.Post.Options.FaceAlpha)
        end

    else
        if fem.Post.Options.ShowEdge
            patch('faces',tria,...
                 'vertices', node,...
                 'edgecolor',fem.Post.Options.ColorEdge,...
                 'facecolor','none',...
                 'parent',fem.Post.Options.ParentAxes)
        else
           patch('faces',tria,...
                 'vertices', node,...
                 'edgecolor','none',...
                 'facecolor','none',...
                 'parent',fem.Post.Options.ParentAxes)
        end
                
    end

end


% PLOT QUAD:
nquad=fem.Sol.Quad.Count;

if nquad>0

    idnode=unique(fem.Sol.Quad.Element(:));   
    node=fem.xMesh.Node.Coordinate(idnode,:);

    quad=renumberElements(fem.Sol.Quad.Element, idnode);
    
    model.Quad.Face=quad;
    model.Quad.Node=node;
    model.Quad.Trasparency = 1 - fem.Post.Options.FaceAlpha;
    model.Quad.Color=fem.Post.Options.ColorPatch;
    model.Quad.Shade='uniform';
    
    if fem.Post.Options.ShowPatch
        if fem.Post.Options.ShowEdge
            patch('faces',quad,...
                 'vertices',node,...
                 'edgecolor',fem.Post.Options.ColorEdge,...
                 'facecolor',fem.Post.Options.ColorPatch,...
                 'parent',fem.Post.Options.ParentAxes,...
                 'facealpha',fem.Post.Options.FaceAlpha)
        else
            patch('faces',quad,...
                 'vertices',node,...
                 'edgecolor','none',...
                 'facecolor',fem.Post.Options.ColorPatch,...
                 'parent',fem.Post.Options.ParentAxes,...
                 'facealpha',fem.Post.Options.FaceAlpha)
        end
        
    else
        if fem.Post.Options.ShowEdge
            patch('faces',quad,...
                 'vertices',node,...
                 'edgecolor',fem.Post.Options.ColorEdge,...
                 'facecolor','none',...
                 'parent',fem.Post.Options.ParentAxes)
        else
           patch('faces',quad,...
                 'vertices',node,...
                 'edgecolor','none',...
                 'facecolor','none',...
                 'parent',fem.Post.Options.ParentAxes)
        end
        
    end

end

%--
if nargout~=0                  
    varargout{1}=model;
end

%
view(3)
axis equal

if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end

