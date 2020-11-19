% initialise database structure
function database=initDatabase()

% INPUT
database.Input=[];

% MODEL
database.Model.Nominal=femInit();

% variation using CoP
database.Model.Variation.Option.NoCluster=5; % no of cluster
database.Model.Variation.Option.MaxIter=3; % no of smooth iterations
database.Model.Variation.Option.SubSampling=0.5; % down sampling percentage
% variation using morphing mesh
database.Model.Variation.Option.SearchDist=10.0; % normal distance
database.Model.Variation.Option.NoSample=1; % no. of sample

% Global setting on dimple
database.Model.Setting.Dimple.Delta=2.0;
database.Model.Setting.Dimple.Height=0.2;
database.Model.Setting.Dimple.Length=3.0;
database.Model.Setting.Dimple.SearchDist=[10.0 5.0]; % normal and tangential
database.Model.Setting.Dimple.Layout={6,'Layout [1]', 'Layout [2]', 'Layout [3]', 'Layout [4]', 'Layout [5]', 'No Dimple'};

% global setting for mesh repair
database.Model.Setting.Repair.SearchDist=0.5;

% SOLUTION
database.Assembly=initAssemblyDatabase();
