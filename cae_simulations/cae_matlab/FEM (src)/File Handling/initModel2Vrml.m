% define intial setting for VRML input database
function model=initModel2Vrml()

% init TRIA
model.Tria.Face=[]; % faces
model.Tria.Node=[]; % nodes
model.Tria.Data=[]; % data for color plotting
model.Tria.Animation=false; % data for color animation
model.Tria.Shade='uniform'; % uniform/none/interp
model.Tria.Trasparency=0.0; % trasparency (only if "uniform" is used)
model.Tria.Color=[1 0 0]; % color (only if "uniform" is used)

%--
model.Tria.ShapeId=[];
model.Tria.MaterialId=[];
model.Tria.ColorId=[];
model.Tria.SensorId=[];

% init QUAD
model.Quad.Face=[];
model.Quad.Node=[];
model.Quad.Data=[];
model.Quad.Animation=false;
model.Quad.Shade='uniform'; % uniform/none/interp
model.Quad.Trasparency=0.0;
model.Quad.Color=[1 0 0];

%--
model.Quad.ShapeId=[];
model.Quad.MaterialId=[];
model.Quad.ColorId=[];
model.Quad.SensorId=[];

% init Text
model.Text.String=[];
model.Text.Position=[0 0 0];
model.Text.FontSize=12;
model.Text.Trasparency=0.0;
model.Text.Color=[1 0 0];
model.Text.TextId=[];
model.Text.MaterialId=[];
