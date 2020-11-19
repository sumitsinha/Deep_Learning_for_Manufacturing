% Log current selection and activate interactive motion control 
function logCurrentSelection(src, event, logPanel)

% logPanel.
    % Panel.
        % .Object => table object ID
        % .ID => (row,col) ID
    % motionData.
        % .Enable
        % .Fig => current figure
        % .Axes => current axes
        % .Target => target "tag"
        % .Rf => rotation matrix selected object (3x3)
        % .Pf => position of selected object (1x3)
        % .Direction => direction of motion (1/2/3/4/5/6) = [rotx, roty, rotz, dx, dy, dz]
        % .MaxRotation => max rotation
        % .MaxTranslation => max translation

%--
if nargin==2
    return
end

if nargin==3
   if ~isstruct(logPanel) 
       return
   end
end
%
if isempty(logPanel.motionData)
    return
end
%
if ~logPanel.motionData.Enable
    return
end
%
% check Target selection
tag=get(src, 'tag');
if ~strcmp(logPanel.motionData.Target, tag)
    return
else
    % set motion
    set(logPanel.motionData.Fig,'WindowButtondownFcn','');
    set(logPanel.motionData.Fig,'WindowButtonupFcn','');
    set(logPanel.motionData.Fig,'WindowButtonMotionFcn','');
    
    % get axis limits
    camPos=get(logPanel.motionData.Axes,'CameraPosition');
    camTar=get(logPanel.motionData.Axes,'CameraTarget');
    camAngle=get(logPanel.motionData.Axes,'CameraViewAngle');

    % Current mouse selection
    Pp=get(logPanel.motionData.Axes,'CurrentPoint'); %- picked point
    Pps=Pp(1,:);
    %
    obj=findobj(logPanel.motionData.Fig, 'tag',logPanel.motionData.Target);
    nobj=length(obj);
    for i=1:nobj
        if strcmp(get(obj(i),'type'),'patch')
            d=get(obj(i),'Vertices');
        else
            X=get(obj(i),'xdata');
            Y=get(obj(i),'ydata');
            Z=get(obj(i),'zdata');

            d=cell(1,3);
            d{1}=X;
            d{2}=Y;
            d{3}=Z;
        end
        set(obj(i),'UserData',d);
    end

    %- start mouse motion
    set(logPanel.motionData.Fig,'WindowButtonMotionFcn',{@moveMouse, logPanel, Pps, camPos, camTar, camAngle, src})
end
%
% Motion control
function moveMouse(src, event, logPanel, Ps, camPos, camTar, camAngle, srcObj)
%
% Current mouse selection
Pp=get(logPanel.motionData.Axes,'CurrentPoint'); %- picked point
Pp=Pp(1,:);
%    
% run max motion check
if logPanel.motionData.Direction>3 % translation
    Ps=applyinv4x4(Ps, logPanel.motionData.Rf, logPanel.motionData.Pf);
    Pp=applyinv4x4(Pp, logPanel.motionData.Rf, logPanel.motionData.Pf);
    
    if logPanel.motionData.Direction==4
        tmotion=Pp(1)-Ps(1);
    elseif logPanel.motionData.Direction==5
        tmotion=Pp(2)-Ps(2);
    elseif logPanel.motionData.Direction==6
        tmotion=Pp(3)-Ps(3);
    end
    if abs(tmotion)>=logPanel.motionData.MaxTranslation
        return
    end
elseif logPanel.motionData.Direction<=3 % rotation

    Ps=applyinv4x4(Ps, logPanel.motionData.Rf, logPanel.motionData.Pf);
    Pp=applyinv4x4(Pp, logPanel.motionData.Rf, logPanel.motionData.Pf);

    tmotion=-logPanel.motionData.SpeedCameraMotion*dot((Pp(1:2)-Ps(1:2)), [1 0]);
        
    %
    if abs(tmotion)>=logPanel.motionData.MaxRotation*pi/180
        return
    else
    end
end
%
% Find all objects with same tag
obj=findobj(logPanel.motionData.Fig, 'tag',logPanel.motionData.Target);
nobj=length(obj);
%
% Run motion control
for i=1:nobj
    moveObject(obj(i),logPanel.motionData.Direction, tmotion,...
                      logPanel.motionData.Rf, logPanel.motionData.Pf,...
                      logPanel.Panel.Object, logPanel.Panel.ID);
    % set axis limits
    set(logPanel.motionData.Axes,'CameraPosition',camPos);
    set(logPanel.motionData.Axes,'CameraTarget',camTar);
    set(logPanel.motionData.Axes,'CameraViewAngle',camAngle);
end
%
% Call mouse buttonUp
set(logPanel.motionData.Fig,'WindowButtonUpFcn',{@endClick, logPanel, srcObj});

%--------------------------
function endClick(~, ~, logPanel, srcObj)

% disable all controls
set(logPanel.motionData.Fig,'WindowButtondownFcn','');
set(logPanel.motionData.Fig,'WindowButtonupFcn','');
set(logPanel.motionData.Fig,'WindowButtonMotionFcn','');

set(srcObj,'buttondownfcn','');

%--------------------------
function moveObject(obj, parameterType, parameterValue, Rf, Pf, objectTable, tableID)

if strcmp(get(obj,'type'),'patch')
    Pb=get(obj,'UserData');
else   
    d=get(obj,'UserData');
    X=d{1};
    Y=d{2};
    Z=d{3};

    Pb=[X(:),Y(:),Z(:)];
end

Tf=eye(4,4); Tf(1:3,1:3)=Rf; Tf(1:3,4)=Pf; 
[R0w,P0w]=setMotion(parameterType, parameterValue, Tf);
Pa=apply4x4(Pb,R0w,P0w);

pa=Pa(1,:); pa=applyinv4x4(pa,Rf,Pf);
pb=Pb(1,:); pb=applyinv4x4(pb,Rf,Pf);
if parameterType<=3
    va=normaliseV(pa(1:2));
    vb=normaliseV(pb(1:2));
    parameterValueCorrect=acos(dot(va, vb))*180/pi;
    signA=sign(dot(vb-va,[1 0]));
    parameterValue=parameterValueCorrect*signA;
end
%
% Write results
if ~isempty(objectTable) && ~isempty(tableID)
    dTable=get(objectTable,'data');
    dTable{tableID,3}=parameterValue;
    set(objectTable,'data',dTable);
end
%
%... then
if strcmp(get(obj,'type'),'patch')
    set(obj,'Vertices',Pa);
else
    res=size(X);
    X=reshape(Pa(:,1),res(1),res(2));
    Y=reshape(Pa(:,2),res(1),res(2));
    Z=reshape(Pa(:,3),res(1),res(2));
    set(obj,'xdata',X);
    set(obj,'ydata',Y);
    set(obj,'zdata',Z);
end

% Set motion
function [R0w,P0w]=setMotion(parameterType, parameterValue, Tf)

if parameterType==1 % alfa
  Rplc=RodriguesRot([1 0 0],parameterValue);
  Pplc=[0;0;0];
elseif parameterType==2 % beta
  Rplc=RodriguesRot([0 1 0],parameterValue);
  Pplc=[0;0;0]; 
elseif parameterType==3 % gamma
  Rplc=RodriguesRot([0 0 1],parameterValue);
  Pplc=[0;0;0];  
elseif parameterType==4 % deltaX
  Rplc=eye(3,3);
  Pplc=[parameterValue;0;0];    
elseif parameterType==5 % deltaY
  Rplc=eye(3,3);
  Pplc=[0;parameterValue;0];    
elseif parameterType==6 % deltaZ
  Rplc=eye(3,3);
  Pplc=[0;0;parameterValue];    
end
%--
Tplc=eye(4,4);
Tplc(1:3,1:3)=Rplc; Tplc(1:3,4)=Pplc;
T0w=Tf*Tplc*inv(Tf); %#ok<MINV>

% save back
R0w=T0w(1:3,1:3);
P0w=T0w(1:3,4)';    

function V=normaliseV(V)

eps=1e-6;
L=norm(V);

if L<=eps
    L=zeros(length(V));
else
    V=V/L;
end
