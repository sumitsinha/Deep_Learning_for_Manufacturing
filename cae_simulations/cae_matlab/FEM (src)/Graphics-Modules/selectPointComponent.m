% click 3D point
function selectPointComponent(fem, idpart)

% fem: fem structure
% idpart: id part to be selected

%--
% pre-requirement: plot has to be active ("figure" and "axes" have been already loaded)  
%--

if nargin<2
    error('Selection point TOOL: not enought input!') 
end

% define local variable
searchDist=10;
%---------------------

%--
ax=fem.Post.Options.ParentAxes;
fig=get(ax, 'parent');

%--
f=uimenu('Label','Selection TOOL');
uimenu(f,'Label','Select point: geometry','Callback',{@StartClick, fig, ax, fem, idpart, 'point', searchDist});
uimenu(f,'Label','Select point: interpolation','Callback',{@StartClick, fig, ax, fem, idpart, 'interp', searchDist});

%--------------
function StartClick(src, event, fig, ax, fem, idpart, mode, searchDist)

% reset any control
h=rotate3d(fig);
set(h,'enable','off');
h=pan(fig);
set(h,'enable','off');
h=zoom(fig);
set(h,'enable','off');

set(fig,'WindowButtondownFcn','');
set(fig,'WindowButtonupFcn','');
set(fig,'WindowButtonmotionFcn','');

set(fig,'WindowButtondownFcn',{@MouseClick, fig, ax, fem, idpart, mode, searchDist});


% start mouse click
function MouseClick(src, event, fig, ax, fem, idpart, mode, searchDist)

%---------------------
Point = get(ax,'CurrentPoint'); 
Point=Point(1,1:3); % use front click (point closest to the camera position)

CamPos = get(ax,'CameraPosition');
CamTgt = get(ax,'CameraTarget');  

CamDir = CamPos - CamTgt;         

% get the camera view axis
Z_axis = CamDir/norm(CamDir);  

% get projection on the surface
[pInter, flag]=pointNormal2PointProjection(fem, Point, Z_axis, idpart); % project along the camera axis
 
if flag  
    
    if strcmp(mode,'point') % plot the picked point
        
        % plot point
        plot3(pInter(1), pInter(2), pInter(3), 'o', 'parent',ax,'tag','tempobj','markerfacecolor','k','markersize',10)

        % plot label
        s{1,1}=sprintf('X: %f',pInter(1));
        s{2,1}=sprintf('Y: %f',pInter(2));
        s{3,1}=sprintf('Z: %f',pInter(3));
        
        text(pInter(1), pInter(2), pInter(3), char(s),...
             'BackgroundColor','w','tag','tempobj', 'parent',ax)
         
    elseif strcmp(mode,'interp') % plot the interpolated value of the picked point
        
        % interpolation variable
        intepVar=fem.Post.Contour.ContourVariable;

        % id contact pair (if any)
        cpairId=fem.Post.Contour.ContactPair; 

        fem.Post.Interp.InterpVariable=intepVar;
        fem.Post.Interp.Domain=fem.Post.Contour.Domain;
        fem.Post.Interp.ContactPair=cpairId;
        fem.Post.Interp.SearchDist=searchDist;

        % get interpolation
        fem.Post.Interp.Pm=pInter;
        fem=getInterpolationData(fem);
        
        %---
        flagi=fem.Post.Interp.Flag;
        
        if flagi
            ui=fem.Post.Interp.Data;
            
            % plot point
            plot3(pInter(1), pInter(2), pInter(3), 'o', 'parent',ax,'tag','tempobj','markerfacecolor','k','markersize',10)

            % plot label
            s{1,1}=sprintf('X: %f',pInter(1));
            s{2,1}=sprintf('Y: %f',pInter(2));
            s{3,1}=sprintf('Z: %f',pInter(3));
            s{4,1}=sprintf('%s: %f',intepVar, ui);
            
            text(pInter(1), pInter(2), pInter(3), char(s),...
                 'BackgroundColor','w','tag','tempobj', 'parent',ax)
        end
        
    end
    
else
    warning('Selection point TOOL: no point selected!');    
end

%-- deactivate commands
set(fig,'WindowButtondownFcn','','WindowButtonUpFcn','')



