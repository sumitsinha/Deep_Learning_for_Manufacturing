% run nagigation options
function setUpGraphicsNav(camData)

% camData.
    % Figure=figure handle;
    % Axes=axes handle;
    % Light= light handle
    % FrameAxes=frame attached to axes;
    % AxesGCS=GCS axes handle;
    % Option.Sample=dynamic point reduction;
    % Option.Mode="pan", "zoom", "rotate";
    % Option.Visible=true/false =>control the visibility of the axes;
    % Option.Speed=mouse speed
    % Option.EnablePost: true/false => if false only GCS is updated
 
%--------------------------------------------------------------------

%---
obj=get(camData.Axes,'Children'); % total obj on axes
nobj=length(obj);

num_pth=0;
hlight=[];
for i=1:nobj
    if strcmp(get(obj(i),'Type'),'patch') 
        num_pth=num_pth+1;
    elseif strcmp(get(obj(i),'Type'),'light')
        hlight=obj(i);
    end
end

%------
camData.Light=hlight;

c=1;
flag=false;
type_pth=zeros(1,num_pth);
for i=1:nobj
    if strcmp(get(obj(i),'Type'),'patch') 
        flag=true;
        type_pth(c)=i;
        c=c+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~flag  % there is NO PATCH

    camData.Option.EnablePost=false;
    
    % run camera motion  
    runCameraMotion(camData);
            
    if ~isempty(camData.Light)
        camlight(camData.Light,'headlight')
    end

else     % there is at least one patch

    camData.Option.EnablePost=true;
    runCameraMotion(camData);
                  
end

%---
function postClb(camData)

if ~camData.Option.EnablePost
    setGCSframe(camData.AxesGCS, camData.Axes)
    return
end

if ~isempty(camData.Light)
    camlight(camData.Light,'headlight')
end

%set properties of GCS frame
setGCSframe(camData.AxesGCS, camData.Axes)

% start again
setUpGraphicsNav(camData);

%---------
function setGCSframe(haxes3dgcs, haxes3d)

cmpos=get(haxes3d,'Cameraposition');
upvt=get(haxes3d,'CameraUpVector');

%set properties of GCS frame
set(haxes3dgcs,'Cameraposition',cmpos);
set(haxes3dgcs,'CameraUpVector',upvt);

%---------------------------------------------
%---------------------------------------------
function runCameraMotion(camData)

set(camData.Figure,'WindowButtondownFcn','');
set(camData.Figure,'WindowButtonupFcn','');
set(camData.Figure,'WindowButtonmotionFcn','');

set(camData.Figure,'WindowButtondownFcn',{@SclickCam, camData});


% enable mouse down
function SclickCam(~, ~, camData)

Pms=get(camData.Axes,'CurrentPoint'); %- picked point
Pms=Pms(1,:);

%- start mouse motion
set(camData.Figure,'WindowButtonMotionFcn',{@moveMouseCam, camData, Pms})

%--
function moveMouseCam(~, ~, camData, Pms)

Pme=get(camData.Axes,'CurrentPoint'); %- picked point
Pme=Pme(1,:);

if strcmp(camData.Option.Mode, 'zoom')
    runZoonCamera(camData.Axes, Pms, Pme, camData.Option.Speed)
elseif strcmp(camData.Option.Mode, 'pan')
    runPanCamera(camData.Axes, Pms, Pme);
elseif strcmp(camData.Option.Mode, 'rotate')
    runRotateCamera(camData.Axes, Pms, Pme, camData.Option.Speed);    
end

%-call mouse buttonUp
set(camData.Figure,'WindowButtonUpFcn',{@endClickCam, camData});

%--
function endClickCam(~, ~, camData)

% disable all controls
set(camData.Figure,'WindowButtonMotionFcn','');
set(camData.Figure,'WindowButtonUpFcn','');

% run post-operation
postClb(camData);




%--------------------------------------------------------------------------------





%-----------------------------
% ZOOM function (change the view angle)
function runZoonCamera(haxes3d, Pms, Pme, speed)

eps=1e-3;

% read camera
camviewangle=get(haxes3d,'CameraViewAngle');  %- camera angle

% get camera frame
[R0c, Pc]=buildCameraFrame(haxes3d);

Pms=applyinv4x4(Pms, R0c, Pc);
Pme=applyinv4x4(Pme, R0c, Pc);

zoomv=dot((Pme(1:2)-Pms(1:2)), [1 0]) + dot((Pme(1:2)-Pms(1:2)), [0 1]); % X + Y components into camera frame
zoomv=zoomv*speed;

camviewangle=camviewangle - zoomv;

% upper clamp
if camviewangle>=90.0
    camviewangle=90.0-eps;
end

% lower clamp
if camviewangle<=0.0
    camviewangle=eps;
end

% save back
set(haxes3d,'CameraViewAngle', camviewangle);

%-----------------------------
% PAN function (change camera position and camera target parallel to the front plane)
function runPanCamera(haxes3d, Pms, Pme)

% read camera
Pt=get(haxes3d,'CameraTarget');  %- camera target
camviewangle=get(haxes3d,'CameraViewAngle');

% get camera frame
[R0c, Pc]=buildCameraFrame(haxes3d);

Zc=R0c(:,3)';
Zcl=dot((Pt-Pc), Zc);

Pms=applyinv4x4(Pms, R0c, Pc);
Pme=applyinv4x4(Pme, R0c, Pc);

% X and Y components into camera frame
panv(1)=dot((Pme(1:2)-Pms(1:2)), [1 0]); 
panv(2)=dot((Pme(1:2)-Pms(1:2)), [0 1]); 
panv(3)=0.0;

Pc=apply4x4(-panv, R0c, Pc);
Pt=Pc + Zc*Zcl;

% save back
set(haxes3d,'CameraPosition', Pc); 
set(haxes3d,'CameraTarget', Pt); 
set(haxes3d,'CameraViewAngle', camviewangle);

%-----------------------------
% ROTATE function (change camera position and up-vector)
function runRotateCamera(haxes3d, Pms, Pme, speed)

% read camera
Pt=get(haxes3d,'CameraTarget');  %- camera target
camviewangle=get(haxes3d,'CameraViewAngle');
Vupc=get(haxes3d,'CameraUpVector');

% get camera frame
[R0c, Pc]=buildCameraFrame(haxes3d);

Pms=applyinv4x4(Pms, R0c, Pc);
Pme=applyinv4x4(Pme, R0c, Pc);

% X and Y components into camera frame
ry=-speed*dot((Pme(1:2)-Pms(1:2)), [1 0]); 
rx=speed*dot((Pme(1:2)-Pms(1:2)), [0 1]);

% rotation around X
Rx=[1 0       0
    0 cos(rx) -sin(rx)
    0 sin(rx) cos(rx)];

Ry=[cos(ry)  0 sin(ry)
    0        1 0
    -sin(ry) 0 cos(ry)];

% get rotated camera UP vector
Vupc=applyinv4x4(Vupc, R0c, [0 0 0]);
Vupc=apply4x4(Vupc, R0c*Rx*Ry, [0 0 0]);

Pc=applyinv4x4(Pc, R0c, Pt); % frame attached to camera target
Pc=apply4x4(Pc, R0c*Rx*Ry, Pt);

% save back
set(haxes3d,'CameraUpVector', Vupc); 
set(haxes3d,'CameraTarget', Pt); 
set(haxes3d,'CameraPosition', Pc); 
set(haxes3d,'CameraViewAngle', camviewangle);

%--
function [R0c, Pc]=buildCameraFrame(haxes3d)

Pc=get(haxes3d,'CameraPosition'); %- camera position
Pt=get(haxes3d,'CameraTarget');  %- camera target
Vup=get(haxes3d,'CameraUpVector'); %- camera up-vector

%- built frame 
Zc=(Pc-Pt)/norm(Pc-Pt);
Vup=Vup/norm(Vup);

Xc=cross(Vup,Zc);
Xc=Xc/norm(Xc);
Yc=cross(Zc,Xc);
Yc=Yc/norm(Yc);

R0c = [Xc', Yc', Zc']; 

    





















% % run nagigation options
% function setUpGraphicsNav(camData)
% 
% % camData.
%     % Figure=figure handle;
%     % Axes=axes handle;
%     % FrameAxes=frame attached to axes;
%     % AxesGCS=GCS axes handle;
%     % Option.Sample=dynamic point reduction;
%     % Option.Mode="pan", "zoom", "rotate";
%     % Option.Visible=true/false =>control the visibility of the axes;
%     % Option.Speed=mouse speed
%     % Option.EnablePre: true/false
%     % Option.EnablePost: true/false => if false only GCS is updated
%     % Temp.
%         % NewAxes: temp axes
%         % Point: temp points
%         % Light: light handle
%         % VisibleObj: visible obj handle
%  
% % re-ret temp variables
% camData.Temp=[];
% %--------------------------------------------------------------------
% 
% %---
% obj=get(camData.Axes,'Children'); % total obj on axes
% nobj=length(obj);
% 
% num_pth=0;
% hlight=[];
% for i=1:nobj
%     if strcmp(get(obj(i),'Type'),'patch') 
%         num_pth=num_pth+1;
%     elseif strcmp(get(obj(i),'Type'),'light')
%         hlight=obj(i);
%     end
% end
% 
% c=1;
% flag=false;
% type_pth=zeros(1,num_pth);
% for i=1:nobj
%     if strcmp(get(obj(i),'Type'),'patch') 
%         flag=true;
%         type_pth(c)=i;
%         c=c+1;
%     end
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if ~flag  % there is NO PATCH
% 
%     camData.Temp.NewAxes=camData.Axes;
%     camData.Option.EnablePre=false;
%     camData.Option.EnablePost=false;
%     
%     % run camera motion  
%     runCameraMotion(camData);
%             
%     if ~isempty(hlight)
%         camlight(hlight,'headlight')
%     end
% 
% else     % there is at least one patch
% 
%     %get properties of initial axes
%     pos=get(camData.Axes,'CameraTarget');
%     cmpos=get(camData.Axes,'Cameraposition');
%     upvt=get(camData.Axes,'CameraUpVector');
%     camva=get(camData.Axes,'cameraviewangle');
%     posaxes=get(camData.Axes,'pos');
%         %     xliminit=get(camData.Axes,'xlim');
%         %     yliminit=get(camData.Axes,'ylim');
%         %     zliminit=get(camData.Axes,'zlim');
%     
%     %----------------------------------
%     % delete temp objs
%     delete(findobj(camData.Figure,'Tag','copya'));
%     %----------------------------------
% 
%     % Create a new axes
%     newaxes=axes('Parent',camData.FrameAxes,'Units','normalized','Position',posaxes,'Tag','copya', 'visible','off');
% 
%     % set properties of visualization to new axes    
%     set(newaxes,'CameraTarget',pos);
%     set(newaxes,'Cameraposition',cmpos);
%     set(newaxes,'CameraUpVector',upvt);
%     set(newaxes,'cameraviewangle',camva);
%         %     set(newaxes,'xlim',xliminit);
%         %     set(newaxes,'ylim',yliminit);
%         %     set(newaxes,'zlim',zliminit);
%     set(newaxes,'DataAspectRatio',[1 1 1])
%     hold on
% 
%     pv=zeros(1, num_pth);
%     for i=1:num_pth
% 
%         if strcmp(get(obj(type_pth(i)),'visible'),'on')
% 
%             %get vertices of patch
%             vi=get(obj(type_pth(i)),'vertices');
% 
%             % random selection
%             n=size(vi,1);
%             sel = randperm(n);
% 
%             % subs percentage
%             lsample=floor(n*camData.Option.Sample);
% 
%             % safety check
%             if lsample==0
%                 lsample=n;
%             end
% 
%             sel = sel(1:lsample);
% 
%             vi=vi(sel,:);
% 
%             %plot sample points
%             pv(i)=plot3(vi(:,1),vi(:,2),vi(:,3),'.','Parent',newaxes, 'visible', 'off');
% 
%         end
% 
%     end
% 
%     % run camera motion 
%     camData.Temp.NewAxes=newaxes;
%     camData.Temp.Point=pv;
%     camData.Temp.Light=hlight;
%     camData.Option.EnablePre=true;
%     camData.Option.EnablePost=true;
%     
%     runCameraMotion(camData);
%                   
% end
% 
% 
% %-----------
% function camData=preClb(camData)
% 
% % show destination axes
% set(camData.Temp.NewAxes,'visible', 'off');
% set(camData.Temp.Point,'visible','on')
% 
% % hide source axes
% if strcmp(camData.Option.Visible, 'on')
%     set(camData.Axes, 'visible','off')
% end
% 
% obj=get(camData.Axes,'Children');
% 
% % get vis status
% vis_obj=cell(1,length(obj));
% for i=1:length(obj)
%     vis_obj{i}=get(obj(i), 'visible');
% end
% 
% set(obj,'visible','off')
% 
% % save back
% camData.Temp.VisibleObj=vis_obj;
% 
% 
% %---
% function postClb(camData)
% 
% if ~camData.Option.EnablePost
%     setGCSframe(camData.AxesGCS, camData.Axes)
%     return
% end
% 
% % get properties of visualization from newaxes
% pos=get(camData.Temp.NewAxes,'CameraTarget');
% cmpos=get(camData.Temp.NewAxes,'Cameraposition');
% upvt=get(camData.Temp.NewAxes,'CameraUpVector');
% camva=get(camData.Temp.NewAxes,'cameraviewangle');
% posaxes=get(camData.Temp.NewAxes,'pos');
%         % xliminit=get(camData.Temp.NewAxes,'xlim');
%         % yliminit=get(camData.Temp.NewAxes,'ylim');
%         % zliminit=get(camData.Temp.NewAxes,'zlim');
% 
% % delete new axes and sampled points
% delete(camData.Temp.NewAxes);
% 
% set(camData.Axes, 'visible',camData.Option.Visible)
% 
% obj=get(camData.Axes,'Children'); % total obj on axes
% 
% for i=1:length(obj)
%     set(obj(i),'Visible',camData.Temp.VisibleObj{i});
% end
% 
% %set properties of visualization to old axes
% set(camData.Axes,'CameraTarget',pos);
% set(camData.Axes,'Cameraposition',cmpos);
% set(camData.Axes,'CameraUpVector',upvt);
% set(camData.Axes,'cameraviewangle',camva);
% set(camData.Axes,'pos',posaxes);
%         % set(camData.Axes,'xlim',xliminit);
%         % set(camData.Axes,'ylim',yliminit);
%         % set(camData.Axes,'zlim',zliminit);
% 
% if ~isempty(camData.Temp.Light)
%     camlight(camData.Temp.Light,'headlight')
% end
% 
% %set properties of GCS frame
% setGCSframe(camData.AxesGCS, camData.Axes)
% 
% % start again
% setUpGraphicsNav(camData);
% 
% %---------
% function setGCSframe(haxes3dgcs, haxes3d)
% 
% cmpos=get(haxes3d,'Cameraposition');
% upvt=get(haxes3d,'CameraUpVector');
% 
% %set properties of GCS frame
% set(haxes3dgcs,'Cameraposition',cmpos);
% set(haxes3dgcs,'CameraUpVector',upvt);
% 
% %---------------------------------------------
% %---------------------------------------------
% function runCameraMotion(camData)
% 
% set(camData.Figure,'WindowButtondownFcn','');
% set(camData.Figure,'WindowButtonupFcn','');
% set(camData.Figure,'WindowButtonmotionFcn','');
% 
% set(camData.Figure,'WindowButtondownFcn',{@SclickCam, camData});
% 
% 
% % enable mouse down
% function SclickCam(~, ~, camData)
% 
% Pms=get(camData.Temp.NewAxes,'CurrentPoint'); %- picked point
% Pms=Pms(1,:);
% 
% % run pre-operation
% if camData.Option.EnablePre
%     camData=preClb(camData);
% end
% 
% %- start mouse motion
% set(camData.Figure,'WindowButtonMotionFcn',{@moveMouseCam, camData, Pms})
% 
% %--
% function moveMouseCam(~, ~, camData, Pms)
% 
% Pme=get(camData.Temp.NewAxes,'CurrentPoint'); %- picked point
% Pme=Pme(1,:);
% 
% if strcmp(camData.Option.Mode, 'zoom')
%     runZoonCamera(camData.Temp.NewAxes, Pms, Pme, camData.Option.Speed)
% elseif strcmp(camData.Option.Mode, 'pan')
%     runPanCamera(camData.Temp.NewAxes, Pms, Pme);
% elseif strcmp(camData.Option.Mode, 'rotate')
%     runRotateCamera(camData.Temp.NewAxes, Pms, Pme, camData.Option.Speed);    
% end
% 
% %-call mouse buttonUp
% set(camData.Figure,'WindowButtonUpFcn',{@endClickCam, camData});
% 
% %--
% function endClickCam(~, ~, camData)
% 
% % disable all controls
% set(camData.Figure,'WindowButtonMotionFcn','');
% set(camData.Figure,'WindowButtonUpFcn','');
% 
% % run post-operation
% postClb(camData);
% 
% 
% 
% 
% %--------------------------------------------------------------------------------
% 
% 
% 
% 
% 
% %-----------------------------
% % ZOOM function (change the view angle)
% function runZoonCamera(haxes3d, Pms, Pme, speed)
% 
% eps=1e-3;
% 
% % read camera
% camviewangle=get(haxes3d,'CameraViewAngle');  %- camera angle
% 
% % get camera frame
% [R0c, Pc]=buildCameraFrame(haxes3d);
% 
% Pms=applyinv4x4(Pms, R0c, Pc);
% Pme=applyinv4x4(Pme, R0c, Pc);
% 
% zoomv=dot((Pme(1:2)-Pms(1:2)), [1 0]) + dot((Pme(1:2)-Pms(1:2)), [0 1]); % X + Y components into camera frame
% zoomv=zoomv*speed;
% 
% camviewangle=camviewangle - zoomv;
% 
% % upper clamp
% if camviewangle>=90.0
%     camviewangle=90.0-eps;
% end
% 
% % lower clamp
% if camviewangle<=0.0
%     camviewangle=eps;
% end
% 
% % save back
% set(haxes3d,'CameraViewAngle', camviewangle);
% 
% %-----------------------------
% % PAN function (change camera position and camera target parallel to the front plane)
% function runPanCamera(haxes3d, Pms, Pme)
% 
% % read camera
% Pt=get(haxes3d,'CameraTarget');  %- camera target
% camviewangle=get(haxes3d,'CameraViewAngle');
% 
% % get camera frame
% [R0c, Pc]=buildCameraFrame(haxes3d);
% 
% Zc=R0c(:,3)';
% Zcl=dot((Pt-Pc), Zc);
% 
% Pms=applyinv4x4(Pms, R0c, Pc);
% Pme=applyinv4x4(Pme, R0c, Pc);
% 
% % X and Y components into camera frame
% panv(1)=dot((Pme(1:2)-Pms(1:2)), [1 0]); 
% panv(2)=dot((Pme(1:2)-Pms(1:2)), [0 1]); 
% panv(3)=0.0;
% 
% Pc=apply4x4(-panv, R0c, Pc);
% Pt=Pc + Zc*Zcl;
% 
% % save back
% set(haxes3d,'CameraPosition', Pc); 
% set(haxes3d,'CameraTarget', Pt); 
% set(haxes3d,'CameraViewAngle', camviewangle);
% 
% %-----------------------------
% % ROTATE function (change camera position and up-vector)
% function runRotateCamera(haxes3d, Pms, Pme, speed)
% 
% % read camera
% Pt=get(haxes3d,'CameraTarget');  %- camera target
% camviewangle=get(haxes3d,'CameraViewAngle');
% Vupc=get(haxes3d,'CameraUpVector');
% 
% % get camera frame
% [R0c, Pc]=buildCameraFrame(haxes3d);
% 
% Pms=applyinv4x4(Pms, R0c, Pc);
% Pme=applyinv4x4(Pme, R0c, Pc);
% 
% % X and Y components into camera frame
% ry=-speed*dot((Pme(1:2)-Pms(1:2)), [1 0]); 
% rx=speed*dot((Pme(1:2)-Pms(1:2)), [0 1]);
% 
% % rotation around X
% Rx=[1 0       0
%     0 cos(rx) -sin(rx)
%     0 sin(rx) cos(rx)];
% 
% Ry=[cos(ry)  0 sin(ry)
%     0        1 0
%     -sin(ry) 0 cos(ry)];
% 
% % get rotated camera UP vector
% Vupc=applyinv4x4(Vupc, R0c, [0 0 0]);
% Vupc=apply4x4(Vupc, R0c*Rx*Ry, [0 0 0]);
% 
% Pc=applyinv4x4(Pc, R0c, Pt); % frame attached to camera target
% Pc=apply4x4(Pc, R0c*Rx*Ry, Pt);
% 
% % save back
% set(haxes3d,'CameraUpVector', Vupc); 
% set(haxes3d,'CameraTarget', Pt); 
% set(haxes3d,'CameraPosition', Pc); 
% set(haxes3d,'CameraViewAngle', camviewangle);
% 
% %--
% function [R0c, Pc]=buildCameraFrame(haxes3d)
% 
% Pc=get(haxes3d,'CameraPosition'); %- camera position
% Pt=get(haxes3d,'CameraTarget');  %- camera target
% Vup=get(haxes3d,'CameraUpVector'); %- camera up-vector
% 
% %- built frame 
% Zc=(Pc-Pt)/norm(Pc-Pt);
% Vup=Vup/norm(Vup);
% 
% Xc=cross(Vup,Zc);
% Xc=Xc/norm(Xc);
% Yc=cross(Zc,Xc);
% Yc=Yc/norm(Yc);
% 
% R0c = [Xc', Yc', Zc']; 

    

















