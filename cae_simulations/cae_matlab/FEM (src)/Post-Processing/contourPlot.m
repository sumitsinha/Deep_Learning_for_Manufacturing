% countour plot
function varargout=contourPlot(fem, tag)

if nargin==1
    tag='';
end

model=initModel2Vrml();

% domain id
idcomp=fem.Post.Contour.Domain;

% PLOT TRIA:   
opt=getRenderTria(fem,...
                      idcomp);

model.Tria.Face=opt.interpData.Face;
model.Tria.Node=opt.interpData.Node;
model.Tria.Data=opt.interpData.Data;

% PLOT QUAD:   
opt=getRenderQuad(fem,...
                  idcomp);

model.Quad.Face=opt.interpData.Face;
model.Quad.Node=opt.interpData.Node;
model.Quad.Data=opt.interpData.Data;

if ~isempty(model.Tria.Node)

    model.Tria.Shade='interp';
    
    trisurf(model.Tria.Face,...
          model.Tria.Node(:,1),...
          model.Tria.Node(:,2),...
          model.Tria.Node(:,3),...
          'FaceVertexCData',model.Tria.Data,...
          'EdgeColor','none',...
          'FaceAlpha',fem.Post.Options.FaceAlpha,... 
          'FaceColor','interp',... 
          'FaceLighting','phong',...
          'EdgeLighting','phong',...
          'parent',fem.Post.Options.ParentAxes,...
          'tag',tag);

end


if ~isempty(model.Quad.Node)

    model.Quad.Shade='interp';
    
    patch('faces',model.Quad.Face,...
      'vertices',model.Quad.Node,...
      'FaceVertexCData',model.Quad.Data,...
      'EdgeColor','none',... 
      'FaceColor','interp',... 
      'FaceAlpha',fem.Post.Options.FaceAlpha,... 
      'FaceLighting','phong',...
      'EdgeLighting','phong',...
      'parent',fem.Post.Options.ParentAxes,...
      'tag',tag);

end         

if nargout~=0                  
    varargout{1}=model;
end

% %--
% set(fem.Post.Options.ParentAxes,'DataAspectRatio',[1 1 1])
% view(fem.Post.Options.ParentAxes, 3)

if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end


% generate meshgrid on triangle
function [x,y]=meshgridTria(res)

x=[];
y=[];
for i=0:1/res:1
     xi=0:1/res:1-i;
     yi=ones(1,length(xi))*i;
     
     x=[x,xi];
     y=[y,yi];
     
end

% get data for rendering plot
function opt=getRenderTria(fem,...
                           idcomp)

opt.interpData.Face=[];
opt.interpData.Node=[];
opt.interpData.Data=[];

%
intepVar=fem.Post.Contour.ContourVariable;
cpairId=fem.Post.Contour.ContactPair; % id contact pair (if any)

% risoluzione grafica
res=fem.Post.Contour.Resolution;

% fattore di scala
sc=fem.Post.Contour.ScaleFactor;
defflag=fem.Post.Contour.Deformed;

% use nodal solution in case of low resolution (==1)
if res==1
    opt=renderElementLowRes(fem, 'tria', idcomp);
    return
end

%-- additional sampling of the mesh
[csi,eta]=meshgridTria(res);

% get tessellation
tria = delaunay(csi,eta);
nP=length(csi);

nele=length(fem.Domain(idcomp).ElementTria);

for i=1:nele
    
         id=fem.Domain(idcomp).ElementTria(i);
         
         if fem.Options.UseActiveSelection % use selection
             flagactive=fem.Selection.Element.Status(id);
         else
             flagactive=true; % use any element
         end
    
         if flagactive
             
         % get node-element coordinate
         idele=fem.xMesh.Element(id).Element;
         iddofs=fem.xMesh.Element(id).ElementNodeIndex;

         Pil=fem.xMesh.Node.Coordinate(idele,:);

         %.
         if strcmp(intepVar,'gap') % GAP
             
            exprInt=fem.Sol.Gap(cpairId).Gap(idele)';
            
         elseif strcmp(intepVar,'p') % REACTION PRESSURE FORCES
             
            exprInt=getPressureNode(fem, id, cpairId); 
            
         elseif strcmp(intepVar,'user') % user field
             
            exprInt=fem.Sol.UserExp(idele);
            
         end
         
         Ui=[];
         if defflag % deformed frame
           % get displacement field
           Ui(:,1)=fem.Sol.U(iddofs(:,1))'; % u
           Ui(:,2)=fem.Sol.U(iddofs(:,2))'; % v
           Ui(:,3)=fem.Sol.U(iddofs(:,3))'; % w
         end

         % ... from natural space to local real frame
         X=zeros(nP,1);
         Y=zeros(nP,1);
         Z=zeros(nP,1);

         data=zeros(nP,1);
         for k=1:nP

               [N,~]=getNdNtria3node(csi(k),eta(k));

               Pkt(1)=N*Pil(:,1);
               Pkt(2)=N*Pil(:,2);
               Pkt(3)=N*Pil(:,3);

               if defflag % deformed frame
                   X(k,1)=Pkt(1)+dot( N, Ui(:,1) )*sc;
                   Y(k,1)=Pkt(2)+dot( N, Ui(:,2) )*sc;
                   Z(k,1)=Pkt(3)+dot( N, Ui(:,3) )*sc;
               else % un-deformed frame
                   X(k,1)=Pkt(1);
                   Y(k,1)=Pkt(2);
                   Z(k,1)=Pkt(3);
               end

               if strcmp(intepVar,'u')
                   ui=fem.Sol.U(iddofs(:,1)); % u
                   data(k,1)=dot(N, ui);
               elseif strcmp(intepVar,'v')
                   ui=fem.Sol.U(iddofs(:,2)); % v
                   data(k,1)=dot(N, ui);
               elseif strcmp(intepVar,'w')
                   ui=fem.Sol.U(iddofs(:,3)); % w
                   data(k,1)=dot(N, ui);
               elseif strcmp(intepVar,'alfa')
                   ui=fem.Sol.U(iddofs(:,4)); % alfa
                   data(k,1)=dot(N, ui);
               elseif strcmp(intepVar,'beta')
                   ui=fem.Sol.U(iddofs(:,5)); % beta
                   data(k,1)=dot(N, ui);
               elseif strcmp(intepVar,'gamma')
                   ui=fem.Sol.U(iddofs(:,6)); % gamma
                   data(k,1)=dot(N, ui);
               elseif strcmp(intepVar,'gap') || strcmp(intepVar,'p') || strcmp(intepVar,'user') 
                   data(k,1)=dot( N, exprInt );

                   %...........
               end

               % filter data
               if data(k,1)>=fem.Post.Contour.MaxRange
                   data(k,1)=fem.Post.Contour.MaxRange;
               end
               
               if data(k,1)<=fem.Post.Contour.MinRange
                   data(k,1)=fem.Post.Contour.MinRange;
               end
               
         end

         %
         if ~isempty(opt.interpData.Face)
           nnode=max(opt.interpData.Face(:));
         else
           nnode=0;
         end

         tface=tria+nnode;

         % save
         opt.interpData.Face=[opt.interpData.Face
                                        tface];

         tvertex=[X,Y,Z]; % defined into local frame

         % save
         opt.interpData.Node=[opt.interpData.Node
                                         tvertex];

         % save
         opt.interpData.Data=[opt.interpData.Data
                                        data(:)];
                                    
         end
         
end % end "i"


% get data for rendering plot
function opt=getRenderQuad(fem,...
                           idcomp)

opt.interpData.Face=[];
opt.interpData.Node=[];
opt.interpData.Data=[];

%
intepVar=fem.Post.Contour.ContourVariable;
cpairId=fem.Post.Contour.ContactPair; % id contact pair (if any)

% risoluzione grafica
res=fem.Post.Contour.Resolution;

% use nododal solution in case of resolution==1
if res==1
    opt=renderElementLowRes(fem, 'quad', idcomp);
    return
end

% fattore di scala
sc=fem.Post.Contour.ScaleFactor;
defflag=fem.Post.Contour.Deformed;

stepp=2/res;
[csi,eta]=meshgrid(-1:stepp:1,-1:stepp:1);

nele=length(fem.Domain(idcomp).ElementQuad);

for i=1:nele
    
         id=fem.Domain(idcomp).ElementQuad(i);
     
         if fem.Options.UseActiveSelection % use selection
             flagactive=fem.Selection.Element.Status(id);
         else
             flagactive=true; % use any element
         end
    
         if flagactive
             
             % get node-element coordinate
             idele=fem.xMesh.Element(id).Element;
             iddofs=fem.xMesh.Element(id).ElementNodeIndex;

             Pil=fem.xMesh.Node.Coordinate(idele,:);

             %.
             if strcmp(intepVar,'gap') % GAP

                exprInt=fem.Sol.Gap(cpairId).Gap(idele)';

             elseif strcmp(intepVar,'p') % REACTION PRESSURE FORCES

                exprInt=getPressureNode(fem, id, cpairId); 

             elseif strcmp(intepVar,'user') % user field

                exprInt=fem.Sol.UserExp(idele);

             end

             Ui=[];
             if defflag % deformed frame
               % get displacement field
               Ui(:,1)=fem.Sol.U(iddofs(:,1))'; % u
               Ui(:,2)=fem.Sol.U(iddofs(:,2))'; % v
               Ui(:,3)=fem.Sol.U(iddofs(:,3))'; % w
             end

             % ... from natural space to local real frame
             X=zeros(res+1,res+1);
             Y=zeros(res+1,res+1);
             Z=zeros(res+1,res+1);

             data=zeros(res+1,res+1);
             for k=1:res+1
                 for t=1:res+1

                   [N,~]=getNdNquad4node(csi(k,t),eta(k,t));

                   Pkt(1)=N*Pil(:,1);
                   Pkt(2)=N*Pil(:,2);
                   Pkt(3)=N*Pil(:,3);

                   if defflag % deformed frame         
                       X(k,t)=Pkt(1)+dot( N, Ui(:,1) )*sc;
                       Y(k,t)=Pkt(2)+dot( N, Ui(:,2) )*sc;
                       Z(k,t)=Pkt(3)+dot( N, Ui(:,3) )*sc;
                   else % un-deformed frame
                       X(k,t)=Pkt(1);
                       Y(k,t)=Pkt(2);
                       Z(k,t)=Pkt(3);
                   end

                   if strcmp(intepVar,'u')
                       ui=fem.Sol.U(iddofs(:,1)); % u
                       data(k,t)=dot(N, ui);
                   elseif strcmp(intepVar,'v')
                       ui=fem.Sol.U(iddofs(:,2)); % v
                       data(k,t)=dot(N, ui);
                   elseif strcmp(intepVar,'w')
                       ui=fem.Sol.U(iddofs(:,3)); % w
                       data(k,t)=dot(N, ui);
                   elseif strcmp(intepVar,'alfa')
                       ui=fem.Sol.U(iddofs(:,4)); % alfa
                       data(k,1)=dot(N, ui);
                   elseif strcmp(intepVar,'beta')
                       ui=fem.Sol.U(iddofs(:,5)); % beta
                       data(k,1)=dot(N, ui);
                   elseif strcmp(intepVar,'gamma')
                       ui=fem.Sol.U(iddofs(:,6)); % gamma
                       data(k,1)=dot(N, ui);
                   elseif strcmp(intepVar,'gap') || strcmp(intepVar,'p') || strcmp(intepVar,'user') 
                       data(k,t)=dot(N, exprInt);
                       %...........
                   end

                   % filter data (crop to "0")
                   if data(k,t)>=fem.Post.Contour.MaxRange
                       data(k,t)=fem.Post.Contour.MaxRange;
                   end

                   if data(k,t)<=fem.Post.Contour.MinRange
                       data(k,t)=fem.Post.Contour.MinRange;
                   end

                 end
             end

             % get tessellation
             pa=surf2patch(X,Y,Z);

             %
             if ~isempty(opt.interpData.Face)
               nnode=max(opt.interpData.Face(:));
             else
               nnode=0;
             end

             tface=pa.faces+nnode;

             % save
             opt.interpData.Face=[opt.interpData.Face
                                            tface];

             tvertex=pa.vertices; % defined into local frame

             % save
             opt.interpData.Node=[opt.interpData.Node
                                             tvertex];

             % save
             opt.interpData.Data=[opt.interpData.Data
                                            data(:)];

         end 
         
end % end "i"


%-
function p=getPressureNode(fem, idele, idcp)

% get node and dofs ids
idnode=fem.xMesh.Element(idele).Element;
iddofs=fem.xMesh.Element(idele).ElementNodeIndex(:,1:3); % only translation

nnode=length(idnode);

p=zeros(nnode,1);
for i=1:nnode
    
    dofs=iddofs(i,:);
    
    % get reaction forces
    R=fem.Sol.RLagrange(idcp).R(dofs);
    
    % get normal to the "deformed" node
    Nn=fem.Sol.DeformedFrame.Node.Normal(idnode(i),:);
    
    % get value
    p(i)=dot( Nn, R );
    
end

% render elements in case of low mesh resolution...
function opt=renderElementLowRes(fem, eletype, idcomp)

% set initial output
opt.interpData.Face=[];
opt.interpData.Node=[];
opt.interpData.Data=[];
     
% magnification option
defflag=fem.Post.Contour.Deformed;

% magnification factor
sc=fem.Post.Contour.ScaleFactor;

% interpolation variable
intepVar=fem.Post.Contour.ContourVariable;

% id contact pair (if any)
cpairId=fem.Post.Contour.ContactPair; 

%----
if strcmp(eletype,'tria')
    nele=length(fem.Domain(idcomp).ElementTria);
    eles=zeros(nele,3);
elseif strcmp(eletype,'quad')
    nele=length(fem.Domain(idcomp).ElementQuad);
    eles=zeros(nele,4);
end

for i=1:nele
    if strcmp(eletype,'tria')
      idi=fem.Domain(idcomp).ElementTria(i);
    elseif strcmp(eletype,'quad')
      idi=fem.Domain(idcomp).ElementQuad(i);    
    end
    
    eles(i,:)=fem.xMesh.Element(idi).Element;
end

% check active elements
if fem.Options.UseActiveSelection % use selection    
    if strcmp(eletype,'tria')
         ide=fem.Domain(idcomp).ElementTria;
    elseif strcmp(eletype,'quad')
         ide=fem.Domain(idcomp).ElementQuad;
    end
    
    eles=eles(fem.Selection.Element.Status(ide),:);
end

if ~isempty(eles)
    
    idnode=unique(eles(:));   
    node=fem.xMesh.Node.Coordinate(idnode,:);
    nnode=size(node,1);

    eles=renumberElements(eles, idnode);

    iddofs=fem.xMesh.Node.NodeIndex(idnode,:);
    
     %.
     if strcmp(intepVar,'gap') % GAP

        exprInt=fem.Sol.Gap(cpairId).Gap(idnode);

     elseif strcmp(intepVar,'p') % REACTION PRESSURE FORCES

%         exprInt=getPressureNode(fem, id, cpairId); 

     elseif strcmp(intepVar,'user') % user field

        exprInt=fem.Sol.UserExp(idnode);

     end
     
     Ui=zeros(nnode,3);
     if defflag % deformed frame
           % get displacement field
           Ui(:,1)=sc * fem.Sol.U(iddofs(:,1))'; % u
           Ui(:,2)=sc * fem.Sol.U(iddofs(:,2))'; % v
           Ui(:,3)=sc * fem.Sol.U(iddofs(:,3))'; % w
     end
      
     if strcmp(intepVar,'u')
           data=fem.Sol.U(iddofs(:,1)); % u
       elseif strcmp(intepVar,'v')
           data=fem.Sol.U(iddofs(:,2)); % v
       elseif strcmp(intepVar,'w')
           data=fem.Sol.U(iddofs(:,3)); % w
       elseif strcmp(intepVar,'alfa')
           data=fem.Sol.U(iddofs(:,4)); % alfa
       elseif strcmp(intepVar,'beta')
           data=fem.Sol.U(iddofs(:,5)); % beta
       elseif strcmp(intepVar,'gamma')
           data=fem.Sol.U(iddofs(:,6)); % gamma
       elseif strcmp(intepVar,'gap') || strcmp(intepVar,'p') || strcmp(intepVar,'user') 
           data=exprInt;

           %...........
     end
          
     % save
     opt.interpData.Node=node+Ui;
     
     % filter elements
     eles=filterFace(eles, [fem.Post.Contour.MinRangeCrop fem.Post.Contour.MaxRangeCrop], data);    
     opt.interpData.Face=eles;
          
     % filter data
     data(data>=fem.Post.Contour.MaxRange)=fem.Post.Contour.MaxRange;
     data(data<=fem.Post.Contour.MinRange)=fem.Post.Contour.MinRange;
     
     % save
     if size(data,2)>1
         data=data';
     end
     
     opt.interpData.Data=data;
         
end

%--------
function eless=filterFace(eles, ranges, data)

% ranges=[min max]
eless=[];

% get counter
count=0;

[n, m]=size(eles); 
for i=1:n
     elesi=eles(i,:);

     flag=true;
     for j=1:m
        if data(elesi(j)) >= ranges(2) ||  data(elesi(j)) <= ranges(1)
            flag=false;
            break
        end
     end

     if flag
         count=count+1;
     end

end

% allocate elements
if count>0

   eless=zeros(count,m);
   
   c=1;
   for i=1:n
         elesi=eles(i,:);

         flag=true;
         for j=1:m
            if data(elesi(j)) >= ranges(2) ||  data(elesi(j)) <= ranges(1)
                flag=false;
                break
            end
         end

         if flag
             eless(c,:)=elesi;
             c=c+1;
         end
         
   end
   
end


