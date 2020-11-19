%--
% This script simulation of door halo with placement error

function halo_reinforcement_learning(run_id,type_flag)
% Define input training model parameters
% if ~exist('run_id','var'), run_id=0; end
% disp(run_id);

if ~exist('type_flag','var'), type_flag='train'; end
disp(type_flag);
%% Set VRM Path in Runtime
% install source codes
fprintf('     Installing src code...\n');
fem_src=fullfile(cd,'../../../../FEM (src)');
if ~exist(fem_src,'dir')
    error('Installation of VRM (error): failed to locate "FEM(src)" folder!')
end
addpath(genpath(fem_src));
%
fao_src=fullfile(cd,'../../../../FAO (src)');
if ~exist(fao_src,'dir')
    error('Installation of VRM (error): failed to locate "FAO(src)" folder!')
end
addpath(genpath(fao_src));
%
% install demo folder
fprintf('     Installing Demo pack...\n');
demos_src=fullfile(cd,'../../../../Demos');
if ~exist(demos_src,'dir')
    error('Installation of VRM (error): failed to locate "Demos" folder!')
end
addpath(genpath(demos_src));
%
% install GUI
fprintf('     Installing GUI...\n');
gui_src=fullfile(cd,'../../../../GUI');
if ~exist(gui_src,'dir')
    error('Installation of VRM (error): failed to locate "GUI" folder!')
end
addpath(genpath(gui_src));

fprintf('Installation completed!\n');
%% Input Output Files
output_path='C:\Users\SINHA_S\Desktop\cross_member_datagen\Demos\Fixture simulation\Multi station\locator_halo\ddpg_data';
file_name='state_t_';
run_id_str=num2str(run_id);
ai_path=append(output_path,'\',file_name,run_id_str,'.csv');
disp(ai_path);
filenameX=fullfile(findfile(cd, ai_path),ai_path);

X_from_AI = importdata(filenameX);% [mm,mm,mm,mm,degree,mm,mm,...,mm] #12 process parameters

% %
% % Convert the Angle (parameter(5)) in radians
X_from_AI(:,1)=X_from_AI(:,1)*pi/180;
X_from_AI
%% Define input files
%
%   Mesh file
%
mhfile{1}=fullfile(findfile(cd, 'part[1]_multi_station_3.inp'),'part[1]_multi_station_3.inp');
%
%   ClampS file
%
clampsfile=fullfile(findfile(cd, 'clampS_multi_station_3.txt'),'clampS_multi_station_3.txt');
%
%
% PinHole file
%
pinholefile=fullfile(findfile(cd, 'pinhole_multi_station_3.txt'),'pinhole_multi_station_3.txt');
%
%
% PinSlot file
%
pinslotfile=fullfile(findfile(cd, 'pinslot_multi_station_3.txt'),'pinslot_multi_station_3.txt');
%
%% Define material properties
% * Thickness
% * Young's Modulus
% * Poisson's ratio
Th=[2.2];
E=[69e3];
nu=[0.3];
%
%% Define stations
idStation=1; % locator placement
stationData(idStation)=initStation();
stationData(idStation).Part=[1];
stationData(idStation).PinHole=[1];
stationData(idStation).PinSlot=[1];
stationData(idStation).Type{1}=2;  
%
idStation=2; % clamp
stationData(idStation)=initStation();
stationData(idStation).Part=[1];
stationData(idStation).PinHole=[1];
stationData(idStation).PinSlot=[1];
stationData(idStation).ClampS=[1 2 3 4]; 
stationData(idStation).Type{1}=3;  
%
idStation=3; % release
stationData(idStation)=initStation();
stationData(idStation).Part=[1];
stationData(idStation).PinHole=[1];
stationData(idStation).PinSlot=[1];
stationData(idStation).ClampS=[1 2]; 
stationData(idStation).Type{1}=4;  
%
%% Initialize model
model=initModel();

% Solver settings
model.database.Assembly.Solver.UseParallel=false; % enable/disable parellel pool
showResults=false; % show/hide visualisation of results
%
%% Define parts and properties
nmesh=length(mhfile);
for i=1:nmesh
    model.database=modelAddItem(model.database, 'Part');
    model.database.Input.Part(i).E=E(i);
    model.database.Input.Part(i).nu=nu(i);
    model.database.Input.Part(i).Th=Th(i);
    model.database.Input.Part(i).Mesh{1}=mhfile{i};
    
    model.database.Input.Part(i).Graphic.ShowEdge=true;
    model.database.Input.Part(i).Graphic.FaceAlpha=0.0;
    model.database.Input.Part(i).Placement.ShowFrame=true;
end
model.database.Input.Part(1).Parametrisation.UCS=1;
model.database.Input.Part(1).Parametrisation.Type(3)=1; % Model training paramater[1] - rotation around pin
%model.database.Input.Part(1).Parametrisation.Type(4)=1; % Model training paramater[2] - translation along pin
%model.database.Input.Part(1).Parametrisation.Type(5)=1; % Model training paramater[3] - translation along pin
%
%% Define locator layout
%
% Hole
model.database=modelImportInput(model.database, pinholefile, 'Hole');
model.database.Input.PinLayout.Hole.Geometry.Shape.Parameter.Diameter=20;
model.database.Input.PinLayout.Hole.Parametrisation.Geometry.ShowFrame=true;
model.database.Input.PinLayout.Hole.TangentType{1}=1; % user
%
% Slot
model.database=modelImportInput(model.database, pinslotfile, 'Slot');
model.database.Input.PinLayout.Slot.Geometry.Shape.Parameter.Diameter=20;
model.database.Input.PinLayout.Slot.Geometry.Shape.Parameter.Length=20;
model.database.Input.PinLayout.Slot.Parametrisation.Geometry.ShowFrame=false;
%
% ClampS
model.database=modelImportInput(model.database, clampsfile, 'ClampS');
%
% Define parameters
for i=1:length(model.database.Input.Locator.ClampS)
    model.database.Input.Locator.ClampS(i).Parametrisation.Geometry.ShowFrame=true;
    model.database.Input.Locator.ClampS(i).NormalType{1}=1; % "User" 
    model.database.Input.Locator.ClampS(i).Nm=[0 1 0]; 
    model.database.Input.Locator.ClampS(i).NmReset=[0 1 0];
    model.database.Input.Locator.ClampS(i).SearchDist=[100 20];
    model.database.Input.Locator.ClampS(i).Graphic.FaceAlpha=1.0;
end
%
%% Build reference model
model.database=modelBuildPart(model.database,[1 1 1 1 1]); 
model.database=modelBuildInput(model.database, [1 0 0]);
%
%% Compute and assign parameters
model.database=modelGetParameters(model.database);
%
%% Run sampling
model.database.Assembly.SamplingStrategy{1}=3;
model.database=modelAddItem(model.database, 'Parameter');
model.database.Input.Parameter(1).X=X_from_AI;
model.database.Assembly.SamplingOptions.IdTable=1; % Parameter table
npara=size(X_from_AI,2);
model.database.Assembly.Group=zeros(npara,4); % [group ID, min, max, resolution]
model.database.Assembly.Group(1:end,1)=1:npara;
for i=1:npara
    model.database.Assembly.Parameter{i,2}=i;
end
%
model.database=modelSampleParameters(model.database);
%
%% Run simulation
model.database.Assembly.Solver.Eps=0.5;
model.database.Assembly.Solver.MaxIter=100;
model.database.Assembly.Solver.LinearSolver{1}=2;
model.database.Assembly.Solver.PenaltyStiffness=1e9;
model.database.Assembly.Solver.UseSoftSpring=false;
model.database.Assembly.Solver.SoftSpring=0.2;
%
useparallel=model.database.Assembly.Solver.UseParallel;
%
[nSimulations, nvars]=size(model.database.Assembly.X.Value);


U=cell(1, nSimulations);
inputData=cell(1, nSimulations);
if useparallel % USE PARALLEL MODE
    parfor paraID=1:nSimulations
        [U{paraID},...
         ~,...
         flagSimulation,...
         inputData{paraID}]=modelSolve(model.database, stationData, paraID);  %#ok<PFBNS>
    end
else % USE SEQUENTIAL MODE
    for paraID=1:nSimulations
        [U{paraID},...
         ~,...
         flagSimulation,...
         inputData{paraID}]=modelSolve(model.database, stationData, paraID); 
    end
end
%
% Save back
model.Session.Station=stationData;
model.Session.Input=inputData;
model.Session.U=U;
%
nnode=size(model.database.Model.Nominal.xMesh.Node.Coordinate,1);
nStation=length(stationData);
for stationID=1:nStation
    Dx=zeros(nSimulations, nnode+1); % deviation X
    Dy=zeros(nSimulations, nnode+1); % deviation Y
    Dz=zeros(nSimulations, nnode+1); % deviation Z
        for paraID=1:nSimulations
           %if flagSimulation(stationID, paraID) % solved
             Usp=sum(U{paraID}(:,1:stationID),2);
             Dxp=Usp(1:6:end)';
             Dyp=Usp(2:6:end)';
             Dzp=Usp(3:6:end)';
             
             Dx(paraID, :)=[Dxp, 1];
             Dy(paraID, :)=[Dyp, 1];
             Dz(paraID, :)=[Dzp, 1];
           %end
        end
end

output_path='C:\Users\SINHA_S\Desktop\cross_member_datagen\Demos\Fixture simulation\Multi station\locator_halo\ddpg_data\';
output_x=sprintf('%scop_drl_x_%g.csv',output_path,run_id);
output_y=sprintf('%scop_drl_y_%g.csv',output_path,run_id);
output_z=sprintf('%scop_drl_z_%g.csv',output_path,run_id);

csvwrite(output_x,Dx);
csvwrite(output_y,Dy);
csvwrite(output_z,Dz);

%% Show results
if ~showResults
    return
end

fig=figure('units','normalized','outerposition',[0 0 1 1],...
          'renderer','opengl', 'color','w');
ax=axes;
set(ax,'clipping', 'off','visible','off');
axis equal, hold all, grid on, box on

lighting phong
light
material metal
camproj('perspective')
colorbar
contourcmap('jet')
view([30 24]) 
camzoom(0.7)
%
model.Figure=fig;
model.Axes3D.Axes=ax;
model.Axes3D.Options.ShowAxes=true;
model.Axes3D.Options.LengthAxis=100;
model.Axes3D.Options.SymbolSize=10;

paraID=1;
stationID=1;
contourVar=2;
dataRange=[-1 1];
deformedScale=1;
deformedFlag=true;
animationDelay=0;
animateFlag=false;
opt=1;

render_contour(model,...
                paraID,...
                stationID,...
                contourVar,...
                dataRange,...
                deformedScale,...
                deformedFlag,...
                animationDelay,...
                animateFlag,...
                opt)
            
xlim=get(model.Axes3D.Axes,'xlim');
ylim=get(model.Axes3D.Axes,'ylim');
zlim=get(model.Axes3D.Axes,'zlim');

set(model.Axes3D.Axes,'xlim',xlim)
set(model.Axes3D.Axes,'ylim',ylim)
set(model.Axes3D.Axes,'zlim',zlim)

if length(model.Session.Station)>1
    % model settings
    panv=uipanel('Parent',fig,'Units', 'normalized','Position',[0.02,0.8,0.2,0.1],...
                'Title','Station slider',...
                'FontSize',10);
    uicontrol('Units', 'normalized','style','slider','parent',...
              panv,'Position',[0.05,0.2,0.85,0.5],...
              'Min',0,'Max',length(model.Session.Station)-1,...
              'sliderstep',[1/(length(model.Session.Station)-1) 1/(length(model.Session.Station)-1)],...
              'callback',{@standard_figure_render, model, paraID, contourVar, dataRange, deformedScale, deformedFlag})
end
end