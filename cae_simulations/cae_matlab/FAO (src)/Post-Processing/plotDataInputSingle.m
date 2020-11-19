% plot input data structure
function plotDataInputSingle(data, field, id, paraid, logPanel, tagi)

% data: data structure
% field: field to plot
% id: ID value
% paraid: parameter id
% logPanel: log panel used by the GUI to plot log messages
% tagi: graphical ID for plotting purposes

if nargin<5
    logPanel.Panel=[];
    logPanel.motionData=[];
    tagi='';
end

% get field
[f, flag]=retrieveStructure(data.database, field, id);
if ~flag
    return
end
if ~f.Enable
    return
end

% check status
[flag, idpoint]=checkInputStatus(f, paraid, field);

% define tag for plotting purposes
if nargin<6
    tagi=getTagValue(field, id);
end

%--
if isstruct(logPanel)
    if ~isempty(logPanel.motionData)
        Rf=f.Parametrisation.Geometry.R{idpoint(1)};
        Pf=f.Pm(idpoint(1),:);
        logPanel.motionData.Rf=Rf;
        logPanel.motionData.Pf=Pf;
    end
end
%
for i=1:length(flag)
    if strcmp(field,'ClampS') || strcmp(field,'ClampM') || strcmp(field,'NcBlock')
         plotClampSupport(data, f, field, flag(i), paraid(i), idpoint(i), tagi, logPanel)
    elseif strcmp(field,'Hole') || strcmp(field,'Slot')
         plotHoleSlot(data, f, field, flag(i), paraid(i), tagi, logPanel)
    elseif strcmp(field,'CustomConstraint')
         plotCustomConstraint(data, f, flag(i), idpoint(i), paraid(i), tagi, logPanel)
     elseif strcmp(field,'Dimple')        
         plotDimple(data, f, flag(i), idpoint(i), paraid(i), tagi, logPanel)
    elseif strcmp(field, 'Stitch')
         plotStitch(data, f, flag(i), paraid(i), tagi, logPanel)
    elseif strcmp(field, 'Selection')
         plotSelection(data, f, tagi, logPanel)
    end
    
end

%--
if data.Axes3D.Options.ShowAxes
    set(data.Axes3D.Axes,'visible','on')
else
    set(data.Axes3D.Axes,'visible','off')
end

%------------------------------
       
%-------------
function plotStitch(data, f, flag, id, tag, logPanel)

eps=1e-1;

if flag && f.EnableReset % well calculated
        
    % start
    if f.Parametrisation.Geometry.Type{1}{1}==1 % ref
        ids=1;
    else
       if id>size(f.Pam{1},1)
            ids=size(f.Pam{1},1);
        else
            ids=id;
        end
    end
    
    % end
    if f.Parametrisation.Geometry.Type{2}{1}==1 % ref
        ide=1;
    else
        if id>size(f.Pam{2},1)
            ide=size(f.Pam{2},1);
        else
            ide=id;
        end
    end
        
    if f.Type{1}==1 % linear   
        Pm=[f.Pam{1}(ids,:)
            f.Pam{2}(ide,:)]; 
        rc=3.0;
    elseif f.Type{1}==2 || f.Type{1}==3 % circular; rigid link
        Pm=[f.Pam{1}(ids,:)
            f.Pas{1}(ids,:)]; 
        rc=f.Diameter;
    elseif f.Type{1}==4 % edge
        Pm=[f.Pam{1}(ids,:)
            f.Pam{2}(ide,:)
            f.Pm(3,:)]; 
        [idKnots, flagknots]=boundaryBy3Points(data.database.Model.Nominal, Pm, f.SearchDist(1));
        
        if flagknots~=0
            return
        end
                
        Pm=data.database.Model.Nominal.xMesh.Node.Coordinate(idKnots,:);
        rc=3.0;
    end
    
    % check for part to part gaps
                    %     X=[]; Y=[]; Z=[];
    if f.Type{1}==2 || f.Type{1}==3 % circular; rigid link
                    %         part_to_part_gap=norm(Pm(1,:)-Pm(2,:));
                    %         if part_to_part_gap <= f.Gap       
            Pm=mean(Pm);
            [X,Y,Z]=renderSphereObj(rc, Pm);
                    %         end
    else % linear/edge
        [X, Y, Z]=renderTubeObj(Pm, rc);
    end
       
    if f.Graphic.ShowEdge
        if ~isempty(X)
             patch(surf2patch(X,Y,Z),...
                  'facecolor',f.Graphic.Color,...
                  'edgecolor',f.Graphic.EdgeColor,...
                  'facealpha',f.Graphic.FaceAlpha,...
                  'parent',data.Axes3D.Axes,...
                  'tag', tag,...
                  'buttondownfcn',{@logCurrentSelection, logPanel})
        end
    else
        if ~isempty(X)
          patch(surf2patch(X,Y,Z),...
              'facecolor',f.Graphic.Color,...
              'edgecolor','none',...
              'facealpha',f.Graphic.FaceAlpha,...
              'parent',data.Axes3D.Axes,...
              'tag', tag,...
              'buttondownfcn',{@logCurrentSelection, logPanel})
        end
    end
    
else
    
    Pm=f.Pm;
    plot3(Pm(:,1), Pm(:,2), Pm(:,3),'ko','parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel});
   
    
end

function plotDimple(data, f, flag, idpoint, id, tag, logPanel)

if flag && f.EnableReset % green

    if length(f.Status{idpoint})==1
        id=1;
    end
    
    rc=data.Axes3D.Options.SymbolSize/5;
    Pm=f.Pam{idpoint}(id,:);   

    [X,Y,Z]=renderSphereObj(rc, Pm);

    if f.Graphic.ShowEdge
     patch(surf2patch(X,Y,Z),...
          'facecolor',f.Graphic.Color,...
          'edgecolor',f.Graphic.EdgeColor,...
          'facealpha',f.Graphic.FaceAlpha,...
          'parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel})
    else
      patch(surf2patch(X,Y,Z),...
          'facecolor',f.Graphic.Color,...
          'edgecolor','none',...
          'facealpha',f.Graphic.FaceAlpha,...
          'parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel})
    end
    
end

function plotCustomConstraint(data, f, flag, idpoint, id, tag, logPanel)

if flag && f.EnableReset % green

   if f.Parametrisation.Geometry.ShowFrame
        R=f.Parametrisation.Geometry.R{idpoint};
        P=f.Pm(idpoint,:);

        lsymbol=data.Axes3D.Options.LengthAxis;
        plotFrame(R, P, data.Axes3D.Axes, lsymbol, tag);
   end

   if f.Parametrisation.Geometry.Type{1}{1}==1 % ref
        id=1;
   end
    
    rc=data.Axes3D.Options.SymbolSize/5;
    Pm=f.Pam{idpoint}(id,:);   

    [X,Y,Z]=renderSphereObj(rc, Pm);

    if f.Graphic.ShowEdge
     patch(surf2patch(X,Y,Z),...
          'facecolor',f.Graphic.Color,...
          'edgecolor',f.Graphic.EdgeColor,...
          'facealpha',f.Graphic.FaceAlpha,...
          'parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel})
    else
      patch(surf2patch(X,Y,Z),...
          'facecolor',f.Graphic.Color,...
          'edgecolor','none',...
          'facealpha',f.Graphic.FaceAlpha,...
          'parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel})
    end
    
end

function plotClampSupport(data, f, field, flag, id, idpoint, tag, logPanel)

if strcmp(field,'ClampM')
    plotSingleClampSupport(data, f, flag, id, 'master', 1, idpoint, tag, logPanel)
    plotSingleClampSupport(data, f, flag, id, 'slave', -1, idpoint, tag, logPanel)
elseif strcmp(field,'ClampS')
    plotSingleClampSupport(data, f, flag, id, 'master', 1, idpoint, tag, logPanel)
    plotSingleClampSupport(data, f, flag, id, 'master', -1, idpoint, tag, logPanel)
elseif strcmp(field,'NcBlock')
    plotSingleClampSupport(data, f, flag, id, 'master', -1, idpoint, tag, logPanel)
end
    

%--
function plotSingleClampSupport(data, f, flag, id, partid, sign, idpoint, tag, logPanel)
      
res=30;
if flag && f.EnableReset % green

    if f.Parametrisation.Geometry.ShowFrame
        R=f.Parametrisation.Geometry.R{idpoint};
        P=f.Pm(idpoint,:);

        lsymbol=data.Axes3D.Options.LengthAxis;
        plotFrame(R, P, data.Axes3D.Axes, lsymbol, tag);
    end

    tgeom=f.Geometry.Shape.Type{1};

    if f.Parametrisation.Geometry.Type{1}{1}==1
       id=1;
    end
    
    Nm=f.Nam{idpoint}(id,:); 
    Tm=f.Tam{idpoint}(id,:); 
    Vm=f.Vam{idpoint}(id,:); 
    if strcmp(partid,'master')
        if tgeom==1 || tgeom==2 || tgeom==3 
            Pm=f.Pam{idpoint}(id,:); 
        elseif tgeom==4 || tgeom==5
            Pm=[f.Pam{idpoint}(id,:)
                f.Pam{idpoint+1}(id,:)]; 
        end
    elseif strcmp(partid,'slave')
        if tgeom==1 || tgeom==2 || tgeom==3 
            Pm=f.Pas{idpoint}(id,:); 
        elseif tgeom==4 || tgeom==5
            Pm=[f.Pas{idpoint}(id,:)
                f.Pas{idpoint+1}(id,:)]; 
        end
    end
         
    angle=f.Geometry.Shape.Rotation;
    if tgeom==1  % cylinder
         rc=f.Geometry.Shape.Parameter.D/2; % radius
         L=f.Geometry.Shape.Parameter.L;
        [face, vertex]=renderCylObjSolid(rc, L, Nm, Tm, Vm, Pm, res, angle, sign);
    elseif tgeom==2 % block
         A=f.Geometry.Shape.Parameter.A;
         B=f.Geometry.Shape.Parameter.B;
         L=f.Geometry.Shape.Parameter.L;
        [face, vertex]=renderBlockObjSolid(A, B, L, Nm, Tm, Vm, Pm, angle, sign);
    elseif tgeom==3 % L-shape
         A=f.Geometry.Shape.Parameter.A;
         B=f.Geometry.Shape.Parameter.B;
         C=f.Geometry.Shape.Parameter.C;
         L=f.Geometry.Shape.Parameter.L;
         [face, vertex]=renderLShapeObjSolid(A, B, C, L, Nm, Tm, Vm, Pm, angle, sign);
    elseif tgeom==4  % couple cylinder
         rc=f.Geometry.Shape.Parameter.D/2; % radius
         L=f.Geometry.Shape.Parameter.L;
         
         P1=Pm(1,:); 
         [face1, vertex1]=renderCylObjSolid(rc, L, Nm, Tm, Vm, P1, res, angle, sign);
        
         P2=Pm(2,:); 
         [face2, vertex2]=renderCylObjSolid(rc, L, Nm, Tm, Vm, P2, res, angle, sign);
        
         face=[face1
              face2+max(face1(:))];
          
         vertex=[vertex1
                vertex2];
            
    elseif tgeom==5  % couple prisma
         A=f.Geometry.Shape.Parameter.A;
         B=f.Geometry.Shape.Parameter.B;
         L=f.Geometry.Shape.Parameter.L;
         
         P1=Pm(1,:); 
         [face1, vertex1]=renderBlockObjSolid(A, B, L, Nm, Tm, Vm, P1, angle, sign);
                 
         P2=Pm(2,:);  
         [face2, vertex2]=renderBlockObjSolid(A, B, L, Nm, Tm, Vm, P2, angle, sign);
        
         face=[face1
              face2+max(face1(:))];
          
         vertex=[vertex1
                vertex2];
    end
        
    % render...
    if strcmp(partid,'slave')
        fColor='r';
    else
        fColor=f.Graphic.Color;
    end
    
    if f.Graphic.ShowEdge
     patch('faces',face,'vertices',vertex,...
          'facecolor',fColor,...
          'edgecolor',f.Graphic.EdgeColor,...
          'facealpha',f.Graphic.FaceAlpha,...
          'parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel})
    else
      patch('faces',face,'vertices',vertex,...
          'facecolor',fColor,...
          'edgecolor','none',...
          'facealpha',f.Graphic.FaceAlpha,...
          'parent',data.Axes3D.Axes,...
          'tag', tag,...
          'buttondownfcn',{@logCurrentSelection, logPanel})
    end
    
    % plot work points
    rc=data.Axes3D.Options.SymbolSize/5;
    np=length(f.Pam);    
    for i=1:np
        
        if f.Status{i}==0 % check if the point is properly calculated
            Pm=f.Pam{i}(id,:); 

            [X,Y,Z]=renderSphereObj(rc, Pm);

            if f.Graphic.ShowEdge
             patch(surf2patch(X,Y,Z),...
                  'facecolor',f.Graphic.Color,...
                  'edgecolor',f.Graphic.EdgeColor,...
                  'facealpha',f.Graphic.FaceAlpha,...
                  'parent',data.Axes3D.Axes,...
                  'tag', tag)
            else
              patch(surf2patch(X,Y,Z),...
                  'facecolor',f.Graphic.Color,...
                  'edgecolor','none',...
                  'facealpha',f.Graphic.FaceAlpha,...
                  'parent',data.Axes3D.Axes,...
                  'tag', tag)
            end
        end
    end
    
end


%--
function plotHoleSlot(data, f, field, flag, id, tag, logPanel)

if flag && f.EnableReset % green

    if f.Parametrisation.Geometry.ShowFrame
        R=f.Parametrisation.Geometry.R{1};
        P=f.Pm(1,:);

        lsymbol=data.Axes3D.Options.LengthAxis;
        plotFrame(R, P, data.Axes3D.Axes, lsymbol, tag);
    end

    % parameter type
    paratype=f.Parametrisation.Geometry.Type{1}{1};
    if paratype==1 || paratype==9 || paratype==10 
        id=1;
    end
      
    Pm=f.Pam{1}(id,:);   
    rc=f.Geometry.Shape.Parameter.Diameter/2;

    % plot--
    if strcmp(field, 'Hole')
        
        Nh=f.Nam{1}(id,:);  
        
        L=2*rc;
        [X,Y,Z]=renderCylObj(rc, -L/2, L/2, Nh, Pm);
        
        if f.Graphic.ShowEdge
         patch(surf2patch(X,Y,Z),...
              'facecolor',f.Graphic.Color,...
              'edgecolor',f.Graphic.EdgeColor,...
              'facealpha',f.Graphic.FaceAlpha,...
              'parent',data.Axes3D.Axes,...
              'tag', tag,...
              'buttondownfcn',{@logCurrentSelection, logPanel})
        else
          patch(surf2patch(X,Y,Z),...
              'facecolor',f.Graphic.Color,...
              'edgecolor','none',...
              'facealpha',f.Graphic.FaceAlpha,...
              'parent',data.Axes3D.Axes,...
              'tag', tag,...
              'buttondownfcn',{@logCurrentSelection, logPanel})
        end
      
    elseif strcmp(field, 'Slot')
        
        % calculate rotation matrix        
        Rc=f.Parametrisation.Geometry.R{1};  

        % plot slot    
        L=f.Geometry.Shape.Parameter.Length;
        [X,Y,Z]=renderSlotObj(rc,L, Rc(:,1)', Rc(:,2)', Pm);
        
        if f.Graphic.ShowEdge
         patch(surf2patch(X,Y,Z),...
              'facecolor',f.Graphic.Color,...
              'edgecolor',f.Graphic.EdgeColor,...
              'facealpha',f.Graphic.FaceAlpha,...
              'parent',data.Axes3D.Axes,...
              'tag', tag,...
              'buttondownfcn',{@logCurrentSelection, logPanel})
        else
          patch(surf2patch(X,Y,Z),...
              'facecolor',f.Graphic.Color,...
              'edgecolor','none',...
              'facealpha',f.Graphic.FaceAlpha,...
              'parent',data.Axes3D.Axes,...
              'tag', tag,...
              'buttondownfcn',{@logCurrentSelection, logPanel})
        end
      
    end

end


