% Export simulation dataset (if any of the inputs is "empty" than the data stream wont be saved)
function modelExportDataset(file_model,...
                            file_input,...
                            file_station,...
                            file_U,...
                            file_gap,...
                            file_flag,...
                            file_x, file_y, file_z,...
                            file_Para,...
                            file_statusPara,...
                            file_AIPara,...
                            model, opt)

% Inputs 
% file_ = filenames [nStation]
    % For each station save the following data streams:
        % model: master model
        % inputData: [nSimulation] x [nStation]
        % stationData: [nStation]
        % U: [nSimulation] x [nDoFs x nStation]
        % GAP: [nSimulation] x [nStation]
        % Para: [1 x nParameters]
        % FLAG: [nSimulation x nStation]
        %--
        % Dx (deviation X): [nSimulations, nNode+1] - last column is the "flag" of the simulation (failed/computed)
        % Dy (deviation Y): [nSimulations, nNode+1] 
        % Dz (deviation Z): [nSimulations, nNode+1]
        % modelParaStatus: [nStation] x [nSimulation x nPara]
        %--
        % nSimulations: no. of sampled points for AI training
        % nStation: no. of stations
        % nNode: no. of node in the mesh model
        % nDof: no. of nDof in the mesh model
        % nParameters: no. of parameters in the model
% opt
    % (1) 1/2 => use .Session / .Simulation
    % (2) 0/1 => model
    % (3) 0/1 => input
    % (4) 0/1 => station
    % (5) 0/1 => U
    % (6) 0/1 => Gap
    % (7) 0/1 => Flag
    % (8) 0/1 => Dx/Dy/Dz
    % (9) 0/1 => Parameters list
    % (10) 0/1 => statusPara
%-------------
if isempty(model)
    return
end
inputData=[];
stationData=[];
U=[];
GAP=[];
FLAG=[];
%--
if opt(1)==1
    inputData=model.Session.Input;
    stationData=model.Session.Station;
    U=model.Session.U;
    GAP=model.Session.Gap;
    FLAG=model.Session.Status;
elseif opt(1)==2
    inputData=model.Simulation.Input;
    stationData=model.Simulation.Station;
    U=model.Simulation.U;
    GAP=model.Simulation.Gap;
    FLAG=model.Simulation.Status;
end
%
% MODEL
if opt(2)
    if ~isempty(file_model) && ~isempty(model)
        save(file_model,'model');
    end
end
%
% INPUT
if opt(3)
    if ~isempty(file_input) && ~isempty(inputData)
        save(file_input,'inputData');
    end
end
%
% STATION
if opt(4)
    if ~isempty(file_station) && ~isempty(stationData)
        save(file_station,'stationData');
    end
end
%
% U
if opt(5)
    if ~isempty(file_U) && ~isempty(U)
        save(file_U,'U');
    end
end
%
% GAP
if opt(6)
    if ~isempty(file_gap) && ~isempty(GAP)
        save(file_gap,'GAP');
    end
end
%
% FLAG
if opt(7)
    if ~isempty(file_flag) && ~isempty(FLAG)
        save(file_flag,'FLAG');
    end
end
%
%
% Dx/Dy/Dz
if opt(8)
    if ~isempty(U) && ~isempty(file_x) && ~isempty(file_y) && ~isempty(file_z)
        nSimulations=length(U);
        nStation=size(U{1},2);
        nnode=size(U{1},1)/6; % this is not generic statement. It needs to be improved in the future...
        for stationID=1:nStation
            Dx=zeros(nSimulations, nnode+1); % deviation X
            Dy=zeros(nSimulations, nnode+1); % deviation Y
            Dz=zeros(nSimulations, nnode+1); % deviation Z
            for paraID=1:nSimulations
               if FLAG(paraID, stationID) % solved
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
            csvwrite(file_x{stationID},Dx);
            csvwrite(file_y{stationID},Dy);
            csvwrite(file_z{stationID},Dz);
        end
    end
end
%
% Process parameters
if opt(9)
    if ~isempty(file_Para)
        csvwrite(file_Para,model.database.Assembly.X.Value);
    end
end
%
% Status of process parameters
if opt(10)
    if ~isempty(file_statusPara) && ~isempty(model) && ~isempty(inputData)
        modelParaStatus=modelGetParametersStatus(model.database.Assembly.Parameter,...
                                                 model.database.Assembly.X.Value,...
                                                 inputData);
        % save back
        nStation=size(modelParaStatus,3);
        for stationID=1:nStation
             csvwrite(file_statusPara{stationID}, modelParaStatus(:,:,stationID));
        end
        
        %custom AI input
        if opt(11)
            param_status=modelParaStatus(:,:,nStation);
            X_AI_pv1=param_status(:,1);
            X_AI_pv2=param_status(:,5);
            X_AI_pv3=param_status(:,9);
            X_AI_pv4=param_status(:,12);
            X_AI_pv5=param_status(:,15);
            X_AI_pv6=param_status(:,17);
            X_AI_position=param_status(:,18:26);
            X_AI_joining=param_status(:,27:126);
            X_AI_clamps=param_status(:,127:159);
            
            X_AI_export = cat(2,X_AI_pv1,X_AI_pv2,X_AI_pv3,X_AI_pv4,X_AI_pv5,X_AI_pv6,X_AI_position,X_AI_clamps,X_AI_joining);
            
            % %Convert Rotation into Degrees when saving for AI
            X_AI_export(:,7)=X_AI_export(:,7)*180/pi;
            X_AI_export(:,10)=X_AI_export(:,10)*180/pi;
            X_AI_export(:,13)=X_AI_export(:,13)*180/pi;
            
            csvwrite(file_AIPara, X_AI_export);
        
        end   
    end
end
