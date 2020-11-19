function run_render_model(source, event, h, opt)

% opt=1 =: show
% opt=2 =: hide

data=guidata(h);

if ~data.Session.Flag
    st=get(data.logPanel,'string');
    st{end+1}='Error: failed to load model from current session!';
    set(data.logPanel, 'string',st);
    return
end
%
if opt==1
    reset_rendering(data, 3);
    modelPlotDataGeom(data, 'Part',data.Axes3D.Options.Tag.Model);
elseif opt==2
    reset_rendering(data, 3);
end




