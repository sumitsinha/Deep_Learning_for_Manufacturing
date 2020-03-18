% inizializza le opzioni grafiche
function fem=femPostInit(fem)

% GENERAL GRAPHIC

opt.ParentAxes=[]; % id asse
opt.ShowAxes=true; % visualizza asse

opt.ShowPatch=true; % mostra la patch
opt.ColorPatch='g'; % colore della patch
opt.FaceAlpha=1; % transparency
opt.ShowEdge=true; % mostra gli edge
opt.ColorEdge='k'; % colore degli edge
opt.WidthEdge=10; % spessore della linea
opt.LengthAxis=10; % lunghezza assi
opt.SymbolSize=10; % BC size
opt.SubSampling=0.5; % "%" of sub sampling
opt.LabelSize=5; % size del font del label
opt.ShowProjection=false; % show projected points

% SHOW BOUNDARY

optshowb.BilateralNode=false;% show bilateral constraints defined at node level (true / false)
optshowb.BilateralElement=false; % show bilateral constraints defined at element level (true / false)
optshowb.Unilateral=false; % show unilateral constraints (true / false)
optshowb.PinHole=false; % show pin-hole constraints (true / false)
optshowb.PinSlot=false; % show pin-slot constraints (true / false)
optshowb.RigidLink=false; % show pin-slot constraints (true / false)
optshowb.Dimple=false; % show dimple pairs (true / false)
optshowb.Contact=false; % show contact pairs (true / false)

% SHOW ANNOTATION

optlabel.Domain=1; % domain identification number (integer)
optlabel.Node=false; % show node label (true / false)
optlabel.Element=false; % show element label (true / false)
optlabel.NormalNode=false; % show normal vectors at nodes(true / false)
optlabel.NormalElement=false; % show normal vectors at elements (true / false)

% CONTOUR PLOT

optcont.Domain=1;
optcont.ContourVariable='u'; % "u", "v", "w", "gap",...
optcont.ContactPair=1; % id contact pair
optcont.MaxRange=1e9; % max range plot
optcont.MinRange=-1e9; % min range plot
optcont.MinRangeCrop=-inf;
optcont.MaxRangeCrop=inf;
optcont.Resolution=10; % numero di punti grafici
optcont.Deformed=false;
optcont.ScaleFactor=1; % fattore di scala

% INTERPOLATION

optint.InterpVariable='u'; % "u", "v", "w", "gap",...
optint.ContactPair=1; % id contact pair
optint.SearchDist=1e-6; % search distance
optint.Domain=1;
optint.Pm=[];

optint.Data=[]; 
optint.Pmp=[]; 
optint.Flag=[];

%-------------------------------------------------------------------------
% save all
fem.Post.Options=opt;
fem.Post.ShowBoundary=optshowb;
fem.Post.ShowAnnotation=optlabel;
fem.Post.Contour=optcont;
fem.Post.Interp=optint;



