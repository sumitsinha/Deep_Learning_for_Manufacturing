function load_file_table(source, event, h)

data=guidata(h);

[file, path] = uigetfile({'*.txt';'*.dat';'*.'},'Open database...');
%--
if file>0
    filepath=[path, file];
    
    maxcol=size(data.database.Assembly.X.Value,2);
    checkcol=true;
    p=modelLoadInputFile(filepath, maxcol, checkcol);
    if ~isempty(p)
        %--
        data.Simulation.Parameters=p;
        
        st=get(data.logPanel,'string');
        st{end+1}=sprintf('Message: no. of parameters loaded: %g',maxcol);
        set(data.logPanel, 'string',st);
        
        % save back
        guidata(h, data);
    end
end