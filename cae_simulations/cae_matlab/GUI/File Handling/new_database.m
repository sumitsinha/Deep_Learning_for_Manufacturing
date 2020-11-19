function data=new_database(source, event, h)

data=guidata(h);

choice = questdlg('Would you like to create a new database?', ...
                 'Delete Message', ...
                 'OK',...
                 'Cancel','OK');
     
% Handle response
switch choice
    case 'OK'

    % close menus
    close_menu(source, event, data.setPanel)
        
    % clean rendering windows
    obj=get(data.Axes3D.Axes,'children');
    
    for i=1:length(obj)
       t=get(obj(i),'type');
       if ~strcmp(t, 'light')
           delete(obj(i));
       end
    end

    % data structure
    data.database=initDatabase();  
end

guidata(h, data);
    