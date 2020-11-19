% Rendering functions
function plotSelection(data, f, tag, logPanel)

if nargin==2
    tag='';
    logPanel.Panel=[];
    logPanel.motionData=[];
end

if nargin==3
    logPanel.Panel=[];
    logPanel.motionData=[];
end

% draw prisma
Pc=f.Pm;
Nc1=f.Nm1;
Nc2=f.Nm2;
rc=f.Rm;
 
if f.Type{1}==1 % prisma
    [face, vertex]=renderPrismaObjSolid(rc, Pc, Nc1, Nc2);
elseif f.Type{1}==2 % ellipsoid
    Z=cross(Nc1, Nc2);
    Z=Z/norm(Z);
    Y=cross(Z, Nc1);

    Rc = [Nc1', Y', Z']; 

    [X,Y,Z]=renderEllipsoidObj(rc, Pc, Rc);

    fv=surf2patch(X,Y,Z);

    face=fv.faces;
    vertex=fv.vertices;
end

if f.Graphic.ShowEdge
    patch('faces',face,'vertices',vertex,...
          'facecolor',f.Graphic.Color,...
          'edgecolor',f.Graphic.EdgeColor,...
          'facealpha',f.Graphic.FaceAlpha,...
          'parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel})
else
    patch('faces',face,'vertices',vertex,...
          'facecolor',f.Graphic.Color,...
          'edgecolor','none',...
          'facealpha',f.Graphic.FaceAlpha,...
          'parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel})
end
 
 