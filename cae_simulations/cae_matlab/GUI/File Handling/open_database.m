function open_database(source, event, h)

filepath = uigetdir(pwd,'Select database...');

data=guidata(h);
data.Session.Folder=filepath;
fprintf('Database at: %s\n', filepath);
guidata(h, data);       
