% Pick point from model and plot interpolated variables
function measure_point_contour(data, idpart, intepVar, searchDist, U, tag)

% Read...
fig=data.Figure(1);
ax=data.Axes3D.Axes;

% Reset existing controls
h=rotate3d(fig);
set(h,'enable','off');
h=pan(fig);
set(h,'enable','off');
h=zoom(fig);
set(h,'enable','off');

set(fig,'WindowButtondownFcn','');
set(fig,'WindowButtonupFcn','');
set(fig,'WindowButtonmotionFcn','');

% Run clicking...
set(fig,'WindowButtondownFcn',{@MouseClick, fig,ax, data.database.Model.Nominal,...
                                idpart, intepVar, searchDist, U, tag, data.logPanel,tag});

% start mouse click
function MouseClick(src, event, fig, ax, fem, idpart, intepVar, searchDist, U, tag, logPanel, tagProbe)

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
    
    fem.Sol.U=U;
    fem.Post.Interp.Pm=pInter;
    fem.Post.Interp.Domain=idpart;
    fem.Post.Interp.SearchDist=searchDist;
    fem.Post.Interp.InterpVariable=intepVar;
    fem.Post.Interp.ContactPair=0;
    [~, dintep, flagintep]=getInterpolationData_fast(fem);

    % Plot point       
    if flagintep
        delete(findobj(fig, 'tag',tagProbe))
        plot3(pInter(1), pInter(2), pInter(3), 'o', 'parent',ax,'tag',tag,...
              'markerfacecolor','k','markersize',10)

        st=get(logPanel,'string');
        st{end+1}=sprintf('Interpolated point: [%.3f, %.3f, %.3f]', pInter(1), pInter(2), pInter(3));
        st{end+1}=sprintf('Interpolated variable: %.3f', dintep);
        set(logPanel, 'string',st); 
    end    
else
    st=get(logPanel,'string');
    st{end+1}='Error - no point picked!';
    set(logPanel, 'string',st);   
end

%-- deactivate commands
set(fig,'WindowButtondownFcn','','WindowButtonUpFcn','')



