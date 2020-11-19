function data=initModel()

% data structure
data.database=initDatabase();

% 3D render structure
data.Figure=[];
data.Axes3D=initAxesPlot([]);

% Font size
data.Options.FontSize=12;

% Log panel
data.logPanel=[];

% Table selection
data.Table.Selection=[];

% Session options
data.Session.Folder=-1; % folder path
data.Session.Station=initStation(); % station data
data.Session.Input=[]; % input data
data.Session.U=[];% deviation field
data.Session.Gap=[];% gap field
data.Session.Parameters=[]; 
data.Session.Status=[]; 
data.Session.Flag=false; % loaded/not loaded

% Simulation options
data.Simulation.Station=initStation(); % station data
data.Simulation.Input=[]; % input data
data.Simulation.U=[];% deviation field
data.Simulation.Gap=[];% gap field
data.Simulation.Parameters=[]; 
data.Simulation.Status=[]; 
data.Simulation.Options.MaxRotation=10; % degree
data.Simulation.Options.MaxTranslation=70; % mm
data.Simulation.Flag=false; % loaded/not loaded
data.Simulation.Log=[];
