% Import simulation dataset
function [U, flag]=modelImportDataset(file_x, file_y, file_z, nSimulations, nStation)

% Inputs
% file_x / file_y / file_z = filenames [1 x nStation]
    % For each station save the following data streams:
        % Dx (deviation X): [nSimulations, nNode] 
        % Dy (deviation Y): [nSimulations, nNode] 
        % Dz (deviation Z): [nSimulations, nNode] 
% nSimulations: no. of sampled data points
% nStation: no. of stations
       
% Outputs
% U: deviation field [1xnSimulations][nDof x nStation] 
        % nNode: no. of node in the mesh model
        % nDof: no. of nDof in the mesh model
% flag: true/false => dataset loaded/failed to load
       
%--------------
U=cell(1,nSimulations);
flag=true;

if nStation~=length(file_x) || nStation~=length(file_y) || nStation~=length(file_z)
    flag=false;
    return
end
%--
for stationID=1:nStation
    % load file
    % Dx
    Dx=modelLoadInputFile(file_x{stationID}, []);
    if isempty(Dx)
        flag=false;
        return
    end
    % Dy
    Dy=modelLoadInputFile(file_y{stationID}, []);
    if isempty(Dy)
        flag=false;
        return
    end
    % Dz
    Dz=modelLoadInputFile(file_z{stationID}, []);
    if isempty(Dz)
        flag=false;
        return
    end
    %
    nnode=size(Dx,2);
    %
    % Update "U"
    u=zeros(nnode*6, 1);
    for paraID=1:nSimulations 
        u(1:6:end)=Dx(paraID,:);
        u(2:6:end)=Dy(paraID,:);
        u(3:6:end)=Dz(paraID,:);

        % Save it back
        U{paraID}(:,stationID)=u;    
    end
end
