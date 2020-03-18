% export VRML model
function exportVrmlFile(varargin)

%---------------
% set input data

% set defaults
filename='vrml.wrl';

shape=[];

room.SkyColor=[1 1 1];
         
keyframe=[];

fanimation=false;

% update based on user inputs
if nargin==1
    filename=varargin{1};
elseif nargin==2
    filename=varargin{1};
    shape=varargin{2};   
elseif nargin==3 
    filename=varargin{1};
    shape=varargin{2}; 
    room=varargin{3};
elseif nargin==4
    filename=varargin{1};
    shape=varargin{2}; 
    room=varargin{3};
    keyframe=varargin{4};
elseif nargin==5
    filename=varargin{1};
    shape=varargin{2}; 
    room=varargin{3};
    keyframe=varargin{4};  
    fanimation=varargin{5};
end

%%---------------------------------------------
% preprocessing geometry
Bb=preProcessShape(shape);

% open file
idf=fopen(filename,'w');

% write header
fprintf(idf,'#VRML V2.0 utf8\r\n'); 

fprintf(idf,'WorldInfo {info ["Created by VRM. Copyright: Dr Pasquale Franciosa"]}\r\n'); 
fprintf(idf,'NavigationInfo {type ["EXAMINE" "WALK" "ANY"] headlight TRUE}\r\n'); 
fprintf(idf,'Background {skyColor %.6f %.6f %.6f}\r\n',room.SkyColor(1), room.SkyColor(2), room.SkyColor(3)); 

% create viewpoint
fprintf(idf,'Viewpoint {\r\n'); 
fprintf(idf,'  position %.6f %.6f %.6f\r\n',room.Camera.Position(1), room.Camera.Position(2), room.Camera.Position(3)); 
fprintf(idf,'  orientation %.6f %.6f %.6f %.6f\r\n',room.Camera.Orientation(1), room.Camera.Orientation(2), room.Camera.Orientation(3), room.Camera.Orientation(4)); 
fprintf(idf,'description "my-View"}\r\n'); 

% create object

% get max and min data
[maxdev, mindev]=getMaxMinData(shape);

shape=setShape(idf, shape, Bb.Center, fanimation, maxdev, mindev);

shape=renderText(idf, shape, Bb.Center);

setAnimation(idf, shape, keyframe, fanimation)

% load camera sensor
load protoDatabase

n=length(proto.CameraSensor);
for i=1:n
    fprintf(idf,'%s\r\n',proto.CameraSensor{i}); % 
end






% create animation
%setAnimation(idf, shape, fanimation, colorid, maxdev, mindev)







% % create bounding box
% traspShape=0.5;
% setShape(idf, colShape, traspShape, Bb.Coordinate, Bb.Face, Bb.Center)

fclose(idf); % close file...


% pre-processing geometry
function Bb=preProcessShape(shape)

ns=length(shape);

vert=[];
for i=1:ns
    vert=[vert
       shape{i}.Quad.Node];
   
    vert=[vert
       shape{i}.Tria.Node];
end

% calculate center
Bb.Center=mean(vert);

% calculate Bounding Box
maxx=max(vert(:,1));
maxy=max(vert(:,2));
maxz=max(vert(:,3));

minx=min(vert(:,1));
miny=min(vert(:,2));
minz=min(vert(:,3));

Bb.Size(1)=maxx-minx;
Bb.Size(2)=maxy-miny;
Bb.Size(3)=maxz-minz;

Bb.Coordinate=[minx miny minz
               maxx miny minz
               maxx maxy minz
               minx maxy minz
               minx miny maxz
               maxx miny maxz
               maxx maxy maxz
               minx maxy maxz];

Bb.Face=[1 2 3 4
         5 6 7 8
         1 5 8 4
         2 6 7 3
         1 5 6 8
         4 8 7 3];

% get max and min data
function [maxdev, mindev]=getMaxMinData(shape)

% shape: shape structure

%%
n=length(shape);

maxdev=zeros(1,n);
mindev=zeros(1,n);
for i=1:n
    
    t=[0 0];
    if ~isempty(shape{i}.Quad.Data)
        t(1)=max(shape{i}.Quad.Data);
    end
    
    if ~isempty(shape{i}.Tria.Data)
        t(2)=max(shape{i}.Tria.Data);
    end
       
    maxdev(i)=max(t);
    

    t=[0 0];
    if ~isempty(shape{i}.Quad.Data)
        t(1)=min(shape{i}.Quad.Data);
    end
    
    if ~isempty(shape{i}.Tria.Data)
        t(2)=min(shape{i}.Tria.Data);
    end
       
    mindev(i)=min(t);
    
end

maxdev = max( maxdev );
mindev = min( mindev );


% crate object
function shape=setShape(idf, shape, centerBb, fanimation, maxdev, mindev)

ns=length(shape);

% plot all shapes
for k=1:ns
    
    % render Quads
    shape=renderPatches(idf, shape, k, centerBb, 'Quad', fanimation, maxdev, mindev);
    
    % render Trias
    shape=renderPatches(idf, shape, k, centerBb, 'Tria', fanimation, maxdev, mindev);
          
end

% process TRIA/QUAD
function shape=renderPatches(idf, shape, kshape, centerBb, typ, fanimation, maxdev, mindev)

% idf: file id
% shape: shape structure
% kshape: id of the shape
% centerBb: center of of the Bb
% typ: Quad/Tria
% fanimation: animate data field

%%
shapek=getfield(shape{kshape}, typ);

% process TRIA
face=shapek.Face;
vert=shapek.Node;

if ~isempty(face)

    % options for color
    shade=shapek.Shade;

    shapeid=sprintf('SHAPE_%s_%g',typ, kshape);
    shapek=setfield(shapek, 'ShapeId', shapeid);
    shape{kshape}=setfield(shape{kshape}, typ, shapek);

    % edit face ids
    nnodes=size(face,2);
    face=face-1;

    if nnodes==3 % tria
        face(:,4)=-1;
    elseif nnodes==4 % quad
        face(:,5)=-1;
    end

    fprintf(idf,'\r\n'); 

    fprintf(idf,'DEF %s Transform {\r\n', shapeid); 
    fprintf(idf,'translation %.6f %.6f %.6f,\r\n', -centerBb(1), -centerBb(2), -centerBb(3));

    fprintf(idf,'children [\r\n'); 

    fprintf(idf,'  Shape {\r\n'); 

    matidk=sprintf('MAT_%s_%g',typ, kshape);
    shapek=setfield(shapek, 'MaterialId', matidk);
    shape{kshape}=setfield(shape{kshape}, typ, shapek);
        
    if strcmp(shade,'uniform')
        col=shapek.Color;
        tra=shapek.Trasparency;

        fprintf(idf,'    appearance  Appearance {\r\n'); 
        fprintf(idf,'                material  DEF %s Material {\r\n', matidk); 
        fprintf(idf,'                diffuseColor %.6f %.6f %.6f\r\n', col(1), col(2), col(3));     
        fprintf(idf,'                transparency %.6f\r\n', tra);      
        fprintf(idf,'                                           }\r\n'); % close material        
        fprintf(idf,'                            }\r\n');  % close appearance             
    end

    %-------------
    fprintf(idf,'     geometry IndexedFaceSet {\r\n'); 
    fprintf(idf,'         solid FALSE\r\n'); 
    fprintf(idf,'         coord Coordinate{\r\n');  
    fprintf(idf,'             point [\r\n');  % coordinates    
    fprintf(idf,'             %.6f %.6f %.6f,\r\n',vert');
    fprintf(idf,'                    ]\r\n');  
    fprintf(idf,'                         }\r\n'); 

    if fanimation==false % use static color
        if strcmp(shade,'interp')

            data=shapek.Data;

            % convert data to RGB
            rgb=zeros(length(data),3);
            for i=1:length(data)
                [r g b]=mapRGB(maxdev, mindev, data(i));

                rgb(i,:)=[r g b];
            end

            % define color
            fprintf(idf,'         color Color	{\r\n'); 
            fprintf(idf,'         color [\r\n'); 
            fprintf(idf,'         %.6f %.6f %.6f,\r\n',rgb');
            fprintf(idf,'                ]\r\n');  
            fprintf(idf,'                       }\r\n'); 

        end
    else % use animation color
        if strcmp(shade,'interp')

            coloridk=sprintf('COLOR_%s_%g',typ, kshape);
            shapek=setfield(shapek, 'ColorId', coloridk);
            shape{kshape}=setfield(shape{kshape}, typ, shapek);
                
            % define color
            fprintf(idf,'         color DEF %s Color	{\r\n', coloridk); 
            fprintf(idf,'                       }\r\n'); 
            
        end 
    end

    % define face ids
    if nnodes==3 % tria
        formatFace='         %g, %g, %g, %g,\r\n';
    elseif nnodes==4 % quad
        formatFace='         %g, %g, %g, %g, %g,\r\n';
    end

    fprintf(idf,'         coordIndex [\r\n'); % face ids
    fprintf(idf,formatFace,face');
    fprintf(idf,'                    ]\r\n');   % close face ids
    fprintf(idf,'                              }\r\n');   % close geometry
    fprintf(idf,'         }\r\n');   % close shape
    
    % add a touch sensor
    tsidk=sprintf('TS_%s_%g',typ, kshape);
    shapek=setfield(shapek, 'SensorId', tsidk);
    shape{kshape}=setfield(shape{kshape}, typ, shapek);
    
    fprintf(idf,'DEF %s TouchSensor{}\r\n', tsidk);   % close children
    
    fprintf(idf,'         ]\r\n');   % close children   
    fprintf(idf,'                }\r\n');   % close trasnform

    fprintf(idf,'\r\n'); 

end

% render text shapes
function shape=renderText(idf, shape, centerBb)

nsh=length(shape);

for kshape=1:nsh
    
    shapek=getfield(shape{kshape}, 'Text');

    % process TEXT
    str=shapek.String;

    if ~isempty(str)

        pos=shapek.Position;
        siz=shapek.FontSize;
        col=shapek.Color;
        tra=shapek.Trasparency;

        ns=length(str);

        matid=cell(0,0);
        textid=cell(0,0);
        for i=1:ns

            posi=pos(i,:);

            textidk=sprintf('TEXT_%g_%g',i, kshape);
            textid{end+1}=textidk;

            fprintf(idf,'\r\n'); 

            fprintf(idf,'DEF %s Transform {\r\n', textidk); 
            fprintf(idf,'translation %.6f %.6f %.6f,\r\n', -centerBb(1)+posi(1), -centerBb(2)+posi(2), -centerBb(3)+posi(3));

            fprintf(idf,'children [ Billboard { axisOfRotation 0 0 0 children [\r\n'); 

            fprintf(idf,'  Shape {\r\n'); 
            fprintf(idf,'  appearance Appearance{\r\n'); 
            
            matidk=sprintf('MAT_%g_%g',i, kshape);
            matid{end+1}=matidk;
    
            fprintf(idf,'  material DEF %s Material {diffuseColor %.6f %.6f %.6f transparency %.6f} }\r\n',matidk, col(i, 1),col(i, 2),col(i, 3), tra); 
            fprintf(idf,'  geometry Text {\r\n'); 
            fprintf(idf,'  string ["%s"]\r\n', str{i}); 
            fprintf(idf,'  fontStyle FontStyle {size %.6f style "BOLD"}\r\n', siz(i)); 
            fprintf(idf,'  } } ] } ] }\r\n');  % close     
           

        end
        
            shapek=setfield(shapek, 'TextId', textid);
            shape{kshape}=setfield(shape{kshape}, 'Text', shapek);
        
            shapek=setfield(shapek, 'MaterialId', matid);
            shape{kshape}=setfield(shape{kshape}, 'Text', shapek);

    end

end


% function idinterp=animatePatches(idf, shape, kshape, typ, maxdev, mindev, idinterp)
% 
% shape=getfield(shape{kshape}, typ);
% 
% %--
% flag=shape.Animation;        
% shade=shape.Shade;
% 
% % animate this shape
% if flag
% 
%   if strcmp(shade,'interp')
% 
%       idinterpk=sprintf('ci_%s_%g_color',typ, kshape);
%       
%       fprintf(idf,'DEF %s ColorArrayInterpolator {\r\n', idinterpk); % face ids
%       fprintf(idf,'key [0 1]\r\n'); % face ids
% 
%       data=shape.Data;
% 
%       % convert data to RGB
%       rgb=zeros(length(data)*2,3);
%       rgb(:,3)=1; % use a Blue color as reference
%       for k=1:length(data)
%             [r g b]=mapRGB(maxdev, mindev, data(k));
% 
%             rgb(length(data)+k,:)=[r g b];
%       end
% 
%       fprintf(idf,'keyValue [\r\n'); % open key values
%       fprintf(idf,'         %.6f %.6f %.6f,\r\n',rgb');
%       fprintf(idf,']\r\n');   % close key values
%       fprintf(idf,'}\r\n');   % close interpolator
%       
%        idinterp{end+1}=idinterpk;
% 
%   end
% 
% end
%         
%         
% % set animation based on data field
% function setAnimation(idf, shape, fanimation,colorid, maxdev, mindev)
% 
% % check if animation is active
% if fanimation
%     
%     % load color interpolator
%     load protoDatabase
% 
%     n=length(proto.ColorArrayInterpolator);
%     for i=1:n
%         fprintf(idf,'%s\r\n',proto.ColorArrayInterpolator{i}); % face ids
%     end
%     
%     % define interpolators
%     n=length(shape);
%     
%     idinterp=cell(0,0);
%     for k=1:n
%         idinterp=animatePatches(idf, shape, k, 'Quad', maxdev, mindev, idinterp);
%         idinterp=animatePatches(idf, shape, k, 'Tria', maxdev, mindev, idinterp);
%     end
%     
%     if ~isempty(idinterp)
%         
%         % define timer
%         fprintf(idf,'DEF TS TimeSensor {\r\n'); % --
%         fprintf(idf,'       cycleInterval 5\r\n'); % --
%         fprintf(idf,'       loop TRUE\r\n'); % --
%         fprintf(idf,'                   }\r\n'); % close
% 
%         % define ROUTES
%         n=length(idinterp);
%         
%         for i=1:n
%             fprintf(idf,'ROUTE TS.fraction_changed TO %s.set_fraction\r\n', idinterp{i}); % --
%             fprintf(idf,'ROUTE %s.value_changed TO %s.set_color\r\n', idinterp{i}, colorid{i}); % --
%         end
%                       
%     end
%     
% end






% set animation based on data field
function setAnimation(idf, shape, keyframe, fanimation)

if fanimation
    
    nframe=length(keyframe.Frame);

    for k=1:nframe

        t=keyframe.Frame(k).Time; % time

        % define scalar interpolator
        scintk=sprintf('scalarInt_%g', k);
        if k<nframe
            fprintf(idf,'DEF %s ScalarInterpolator {key	[0, %.6f, %.6f, 1]	keyValue [1, 0, 0, 1]}\r\n', scintk, keyframe.Delay, 1-keyframe.Delay); % --
        else
            fprintf(idf,'DEF %s ScalarInterpolator {key	[0, %.6f, 1]	keyValue [1, 0, 0]}\r\n', scintk, keyframe.Delay); % --
        end
        scint{k}=scintk;

        % define timer
        timeridk=sprintf('timer_%g', k);
        fprintf(idf,'DEF %s TimeSensor {cycleInterval	%.6f	loop	FALSE}\r\n', timeridk, t); % --
        timerid{k}=timeridk;

        % define delays
        if k==1
            counttime=0;
        else
            counttime=counttime+t;
        end

        delayidk=sprintf('delay_%g', k);
        fprintf(idf,'DEF %s Script {eventIn SFTime set_in eventOut SFTime out url "vrmlscript:\r\n', delayidk); % --
        fprintf(idf,'function set_in(t)\r\n'); % --
        fprintf(idf,'out=t + %.6f; " }\r\n', counttime); % --  
        delayid{k}=delayidk;
               
    end

    % define routes
    for k=1:nframe

        triggershape=shape{keyframe.TriggerShape}.Tria.SensorId;
        
        if ~isempty(triggershape)
            fprintf(idf,'ROUTE %s.touchTime TO %s.set_in\r\n', triggershape, delayid{k}); % --
        end
        
        triggershape=shape{keyframe.TriggerShape}.Quad.SensorId;
        
        if ~isempty(triggershape)
            fprintf(idf,'ROUTE %s.touchTime TO %s.set_in\r\n', triggershape, delayid{k}); % --
        end
              
        
        %--------
        fprintf(idf,'ROUTE %s.out TO %s.set_startTime\r\n', delayid{k}, timerid{k}); % --

        fprintf(idf,'ROUTE %s.fraction_changed TO %s.set_fraction\r\n', timerid{k}, scint{k}); % --

        % link to shapes
        nsh=length(keyframe.Frame(k).Shape);
        for j=1:nsh

            ids=keyframe.Frame(k).Shape(j);

            matshape=shape{ids}.Tria.MaterialId;

            if ~isempty(matshape)
                fprintf(idf,'ROUTE %s.value_changed TO %s.set_transparency\r\n', scint{k}, matshape); % --
            end

            matshape=shape{ids}.Quad.MaterialId;

            if ~isempty(matshape)
                fprintf(idf,'ROUTE %s.value_changed TO %s.set_transparency\r\n', scint{k}, matshape); % --
            end
            
            % render text as well
            nt=length(shape{ids}.Text.String);
            
            for z=1:nt
                texshape=shape{ids}.Text.MaterialId{z};
                if ~isempty(texshape)
                    fprintf(idf,'ROUTE %s.value_changed TO %s.set_transparency\r\n', scint{k}, texshape); % --
                end
            end

        end

    end
   
end




