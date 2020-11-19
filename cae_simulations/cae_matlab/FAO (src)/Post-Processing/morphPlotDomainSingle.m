% Plot points and influence domains of morphing mesh model 
function morphPlotDomainSingle(data, idpart, idpoint, logPanel, tag)

% data: input model
% idpart: part ID
% idpoint: point ID

if nargin<4
    logPanel.Panel=[];
    logPanel.motionData=[];
    tag='';
end

if nargin<5
    tag='';
end

% get part structure
[fp,flagp]=retrieveStructure(data.database, 'Part', idpart);

if ~flagp
    warning('Warning (morphing mesh) - part ID not valid!');
    return
end

if isstruct(logPanel)
    if ~isempty(logPanel.motionData)
        [Rc, Pc]=lncRotationPosition(data.database, idpart, idpoint);
         logPanel.motionData.Rf=Rc;
         logPanel.motionData.Pf=Pc;
    end
end

%---
Pc=fp.Morphing(idpoint).Pc;

%--
% Control point
rc=data.Axes3D.Options.SymbolSize;
[X,Y,Z]=renderSphereObj(rc, Pc);
patch(surf2patch(X,Y,Z),...
          'facecolor','g',...
          'edgecolor','k',...
          'facealpha',1,...
          'parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel})

%--
% Influence domain
IDSelection=fp.Morphing(idpoint).Selection;
if IDSelection==0 % use automatic selection
   if fp.Status==0
       idnodes=data.database.Model.Nominal.Domain(idpart).Node;

       nodes=data.database.Model.Nominal.xMesh.Node.Coordinate(idnodes,:);
       dV=getBoundingVolume(nodes);
       dV.Type{1}=2;

       dV.Graphic.FaceAlpha=0.5;
       plotSelection(data, dV, tag, logPanel)
   else
       warning('Warning (morphing mesh) - Part ID not active!');
   end
else % use current selection
   [dV, flag]=retrieveStructure(data.database, 'Selection', IDSelection);

   if flag
        dV.Graphic.FaceAlpha=0.3;
        plotSelection(data, dV, tag, logPanel)
   else
        warning('Warning (morphing mesh) - Selection ID not valid!');
   end
end       


%---
function [Rc, Pc]=lncRotationPosition(data, idpart, idpoint)

%---
seardist=data.Model.Variation.Option.SearchDist;

% control point
Pc=data.Input.Part(idpart).Morphing(idpoint).Pc;

% normal
mnormal=data.Input.Part(idpart).Morphing(idpoint).NormalMode{1};

if mnormal==2 % use model
    [~, Nci, flagi]=point2PointNormalProjection(data.Model.Nominal, Pc, idpart, seardist);

   if ~flagi % use user settings
       error('Error (morphing) - failed to calculate normal vector @ control point [%g]', i)
   end

else % user
   Nci=data.Input.Part(idpart).Morphing(idpoint).Nc; 
end

ln=norm(Nci);

if ln<=eps
    Error('Error (morphing) - failed to calculate normal vector @ control point [%g]', i)
end

Nci=Nci/ln;
Nc=Nci; 

% Then build rotation matrix
Rc=vector2Rotation(Nc);
