function run_load_model(source, event, h)

choice = questdlg('Would you like to load model?', ...
                 'Load Message', ...
                 'OK',...
                 'Cancel','OK');
     
% Handle response
switch choice
    case 'OK'
        data=guidata(h);
        filepath=data.Session.Folder;
        if filepath>0
            data=loadLargeFileGUI(data, filepath);

            reset_rendering(data, 1);
            modelPlotDataGeom(data, 'Part',data.Axes3D.Options.Tag.Model);
            
            guidata(h, data);
            
            st=get(data.logPanel,'string');
            st{end+1}='Message: model loaded from current session!';
            set(data.logPanel, 'string',st);
        else
            st=get(data.logPanel,'string');
            st{end+1}='Error: failed to load model from current session!';
            set(data.logPanel, 'string',st);
        end
end

