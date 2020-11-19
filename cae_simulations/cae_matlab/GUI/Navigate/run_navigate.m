function run_navigate(~, ~, h, opt)
    
data=guidata(h);

% run navigation
if strcmp(opt,'select')
    set(data.Figure(1),'WindowButtondownFcn','');
    set(data.Figure(1),'WindowButtonupFcn','');
    set(data.Figure(1),'WindowButtonmotionFcn',''); 
    
    % delete temp axes
    delete(findobj(data.Figure(1), 'tag', 'copya'))
    return
end

% define camera property
camData.Figure=data.Figure(1);
camData.Axes=data.Axes3D.Axes;
camData.AxesGCS=data.Axes3D.AxesGCS;
camData.Option.Sample=data.Axes3D.Options.SubSampling;
camData.Option.Mode=opt;
camData.Option.Visible=get(data.Axes3D.Axes,'visible');
camData.Option.Speed=data.Axes3D.Options.SpeedCameraMotion; % degree/mouse movement

% run navigation
setUpGraphicsNav(camData)

