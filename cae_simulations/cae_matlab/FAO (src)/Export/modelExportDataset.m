% Export simulation data set
function modelExportDataset(output_x, output_y, output_z, U, FLAG, nnode)

% Inputs
% output_x / output_y / output_z = filenames [1 x nStation]
    % For each station save the following data streams:
        % Dx (deviation X): [nSimulations, nNode+1] - last column is the "flag"
        % Dy (deviation Y): [nSimulations, nNode+1] - last column is the "flag"
        % Dz (deviation Z): [nSimulations, nNode+1] - last column is the "flag"
% U: deviation field [1xnSimulations][nDof x nStation] 
% FLAG: flag [nStation x nSimulations]
        % nSimulations: no. of sampled points for AI training
        % nStation: no. of stations
        % nNode: no. of node in the mesh model
        % nDof: no. of nDof in the mesh model
       
%--------------
nSimulations=length(U);
nStation=size(U{1},2);
for stationID=1:nStation
    Dx=zeros(nSimulations, nnode+1); % deviation X
    Dy=zeros(nSimulations, nnode+1); % deviation Y
    Dz=zeros(nSimulations, nnode+1); % deviation Z
    for paraID=1:nSimulations
       if FLAG(stationID, paraID) % solved
         Usp=sum(U{paraID}(:,1:stationID),2);
         Dxp=Usp(1:6:end)';
         Dyp=Usp(2:6:end)';
         Dzp=Usp(3:6:end)';

         Dx(paraID, :)=[Dxp, 1];
         Dy(paraID, :)=[Dyp, 1];
         Dz(paraID, :)=[Dzp, 1];
       end
    end
    % save back
    csvwrite(output_x{stationID},Dx);
    csvwrite(output_y{stationID},Dy);
    csvwrite(output_z{stationID},Dz);
end

