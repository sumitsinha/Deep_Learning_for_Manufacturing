function export_dataset(source, event, h, opt)

% opt 1: => render session (data.Session)
% opt 2: => render simulation (data.Simulation)

data=guidata(h);

if opt==1
    flagd=data.Session.Flag;
elseif opt==2
    flagd=data.Simulation.Flag;
end
%
if ~flagd
    st=get(data.logPanel,'string');
    st{end+1}='Error: failed to load model from current session!';
    set(data.logPanel, 'string',st);
    return
end
%
% Export
try
    pathdest=data.Session.Folder;
    if opt==1
        nStation=length(data.Session.Station);
    elseif opt==2
        nStation=length(data.Simulation.Station);
    end
    %
    [file_model,...
            file_input,...
            file_station,...
            file_U,...
            file_gap,...
            file_flag,...
            file_x, file_y, file_z,...
            file_Para,...
            file_statusPara]= modelExportDatasetFormatFiles(pathdest, nStation);
    %
    optExport=[opt 0 0 0 0 0 0 1 1 1]; 
    modelExportDataset(file_model,...
                        file_input,...
                        file_station,...
                        file_U,...
                        file_gap,...
                        file_flag,...
                        file_x, file_y, file_z,...
                        file_Para,...
                        file_statusPara,...
                        data, optExport);                      
    %
    st=get(data.logPanel,'string');
    st{end+1}='Message: dataset exported succesfully!';
    set(data.logPanel, 'string',st);
catch
    st=get(data.logPanel,'string');
    st{end+1}='Error: failed to export dataset (check file path)!';
    set(data.logPanel, 'string',st);
end     
