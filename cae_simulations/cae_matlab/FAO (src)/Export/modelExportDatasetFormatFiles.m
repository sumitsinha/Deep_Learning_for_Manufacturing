% Create filenames ready to write dataset on tos
function [file_model,...
            file_input,...
            file_station,...
            file_U,...
            file_gap,...
            file_flag,...
            file_x, file_y, file_z,...
            file_Para,...
            file_statusPara,...
            file_AIPara]= modelExportDatasetFormatFiles(pathdest, nStation)

% Input 
% pathdest: destination folder
% nStation: no. of stations

% Outputs
% file_ = filenames [nStation]
        % model: master model
        % inputData: input data
        % stationData: station data
        % U: assembly simulation
        % GAP: gap data
        % FLAG: simulation flag
        % Dx: deviation X
        % Dy: deviation Y
        % Dz: deviation Z
        % Para: list of process parameters
        % modelParaStatus: status of process parameters
%-----------------

% Export
file_model=[pathdest,'\','model'];
file_input=[pathdest,'\','input'];
file_station=[pathdest,'\','station'];
file_U=[pathdest,'\','U'];
file_gap=[pathdest,'\','GAP'];

file_Para=[pathdest,'\','Parameters.csv'];
file_flag=[pathdest,'\','FLAG'];
%
file_x=cell(1,nStation);
file_y=cell(1,nStation);
file_z=cell(1,nStation);
file_statusPara=cell(1,nStation);
for i=1:nStation
    file_x{i}=sprintf('%s%s%s%g.csv',pathdest,'\','DX_stage_',i);
    file_y{i}=sprintf('%s%s%s%g.csv',pathdest,'\','DY_stage_',i);
    file_z{i}=sprintf('%s%s%s%g.csv',pathdest,'\','DZ_stage_',i);
    
    file_statusPara{i}=sprintf('%s%s%s%g.csv',pathdest,'\','Parameter_status_stage_',i);
end
file_AIPara=[pathdest,'\','AI_Input_Parameters.csv'];
