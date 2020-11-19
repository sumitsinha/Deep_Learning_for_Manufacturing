% load large data structure
function data=loadLargeFileGUI(data, filepath)

%--
fprintf('loading database...\n')
fie={'model','station','input','U', 'GAP', 'FLAG'};

% Model
fprintf('   loading %s...\n', fie{1})
filemat=[filepath, '\', fie{1},'.mat'];
if exist(filemat)
    var=load(filemat);
    data.database=var.model.database;
    data.Session.Flag=true;
else
    error('Loading database - failed to locate "%s"',fie{1});
end
fprintf('   loading %s completed!\n', fie{1})

% Station
fprintf('   loading %s...\n', fie{2})
filemat=[filepath, '\', fie{2},'.mat'];
if exist(filemat)
    var=load(filemat);
    data.Session.Station=var.stationData;
else
    warning('Loading database - failed to locate "%s"',fie{2});
end
fprintf('   loading %s completed!\n', fie{2})

% Input
fprintf('   loading %s...\n', fie{3})
filemat=[filepath, '\', fie{3},'.mat'];
if exist(filemat)
    var=load(filemat);
    data.Session.Input=var.inputData;
else
    warning('Loading database - failed to locate "%s"',fie{3});
end
fprintf('   loading %s completed!\n', fie{3})

% U
fprintf('   loading %s...\n', fie{4})
filemat=[filepath, '\', fie{4},'.mat'];
if exist(filemat)
    var=load(filemat);
    data.Session.U=var.U;
else
    warning('Loading database - failed to locate "%s"',fie{4});
end
fprintf('   loading %s completed!\n', fie{4})

% GAP
fprintf('   loading %s...\n', fie{5})
filemat=[filepath, '\', fie{5},'.mat'];
if exist(filemat)
    var=load(filemat);
    data.Session.Gap=var.GAP;
else
    warning('Loading database - failed to locate "%s"',fie{5});
end
fprintf('   loading %s completed!\n', fie{5})

% FLAG
fprintf('   loading %s...\n', fie{6})
filemat=[filepath, '\', fie{6},'.mat'];
if exist(filemat)
    var=load(filemat);
    data.Session.Status=var.FLAG;
else
    warning('Loading database - failed to locate "%s"',fie{6});
end
fprintf('   loading %s completed!\n', fie{6})
%--
% Set pre-defined values
data.Session.Parameters=data.database.Assembly.X.Value;
data.Simulation.Parameters=data.database.Assembly.X.Value;
data.Simulation.U=data.Session.U;
for i=1:length(data.Simulation.U)
    data.Simulation.U{i}=data.Simulation.U{i}*0.0; % reset deviation field
end
data.Simulation.Flag=data.Session.Flag;
data.Simulation.Input=data.Session.Input;
%--

