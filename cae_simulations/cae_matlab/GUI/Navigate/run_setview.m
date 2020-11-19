% set axes view
function run_setview(~, ~, h, opt)

data=guidata(h);

set(data.Figure(1),'WindowButtondownFcn','');
set(data.Figure(1),'WindowButtonupFcn','');
set(data.Figure(1),'WindowButtonmotionFcn',''); 

haxes3d=data.Axes3D.Axes;
haxes3dgcs=data.Axes3D.AxesGCS;

% set view
if strcmp(opt,'XY')
    view(haxes3d, 2)
    view(haxes3dgcs, 2)
elseif strcmp(opt,'XZ')
    view(haxes3d, [0 0])
    view(haxes3dgcs, [0 0])
elseif strcmp(opt,'YZ')
    view(haxes3d, [90 0])
    view(haxes3dgcs, [90 0])
elseif strcmp(opt,'YX')
    view(haxes3d, [90 -90])
    view(haxes3dgcs, [90 -90])
elseif strcmp(opt,'ZX')
    view(haxes3d, [-180 0])
    view(haxes3dgcs, [-180 0])
elseif strcmp(opt,'ZY')
    view(haxes3d, [-90 0])
    view(haxes3dgcs, [-90 0])
elseif strcmp(opt,'Iso')
    set(haxes3d,'view',[-37.5 30]);
    set(haxes3d,'cameraviewangle',8);
    
    set(haxes3dgcs,'view',[-37.5 30]);
end




