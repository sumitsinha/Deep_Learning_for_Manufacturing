function run_clear_view(source, event, h, opt)

choice = questdlg('Would you like to clear view?', ...
                 'Delete Message', ...
                 'OK',...
                 'Cancel','OK');
     
% Handle response
switch choice
    case 'OK'
        data=guidata(h);
        reset_rendering(data, opt);
end