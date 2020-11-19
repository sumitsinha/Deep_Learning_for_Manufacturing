function export_image(source, event, h)

[file, path] = uiputfile({'*.png';'*.jpg';'*.'},'Open database...');

filep=[path, file];

if file>0
    
    [pathstr, name, ext] = fileparts(filep);
    filepname=[pathstr, '\',name];
    print(['-d',ext(2:end)],'-r300',filepname)
    
end