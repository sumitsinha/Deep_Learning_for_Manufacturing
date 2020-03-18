% plotta la mesh (campo "fem.xMesh")
function model=meshPlotVrml(fem)

fem.Post.Options.ColorPatch=[0.3 0.3 0.3];

model=initModel2Vrml();
    
% PLOT TRIA:
ntria=fem.Sol.Tria.Count;

if ntria>0

    model.Tria.Face=fem.Sol.Tria.Element;
    model.Tria.Node=fem.xMesh.Node.Coordinate;
    model.Tria.Trasparency=0.6;
    model.Tria.Color=fem.Post.Options.ColorPatch;
    model.Tria.Shade='uniform';

end


% PLOT QUAD:
nquad=fem.Sol.Quad.Count;

if nquad>0

    model.Quad.Face=fem.Sol.Quad.Element;
    model.Quad.Node=fem.xMesh.Node.Coordinate;
    model.Quad.Trasparency = 0.6;
    model.Quad.Color=fem.Post.Options.ColorPatch;
    model.Quad.Shade='uniform';

end

