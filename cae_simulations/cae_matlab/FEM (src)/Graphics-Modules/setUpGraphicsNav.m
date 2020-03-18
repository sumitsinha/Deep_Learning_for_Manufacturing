% run nagigation options
function setUpGraphicsNav(camData)

% camData.
    % Figure=figure handle;
    % Axes=axes handle;
    % Light= light handle
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

    






  