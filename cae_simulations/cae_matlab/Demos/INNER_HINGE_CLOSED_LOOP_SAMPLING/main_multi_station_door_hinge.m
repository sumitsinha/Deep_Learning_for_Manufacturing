%---
% This script generates training data for Door inner + hinge reinforcement
%
% INPUTs from AI:
    % X_from_AI (parameters matrix); [nSimulations, nParameters]
% OUTPUTs to AI:
    % Dx (deviation X): [nSimulations, nNode+1, nStation] - last column is the "flag"
    % Dy (deviation Y): [nSimulations, nNode+1, nStation] - last column is the "flag"
    % Dz (deviation Z): [nSimulations, nNode+1, nStation] - last column is the "flag"
        % nSimulations: no. of sampled points for AI training
        % nParameters: no. of training paramaters (KCCs)
            % KKC(1): rotation around pin of hinge
            % KKC(2): translation along pin of hinge
            % KKC(3): translation along pin of hinge
            % KKC(4): translation along N(normal) of clamp[1]
            % KKC(5): translation along N(normal) of clamp[2]
            % KKC(6): translation along N(normal) of clamp[3]
        % nStation: no. of stations
        % nNode: no. of node in the mesh model
        % nDoF: no. of total DoFs in the VRM model
%--
% clc
% clear
% close all
function main_multi_station_door_hinge(run_id,type_flag)

if ~exist('run_id','var'), run_id=0; end
disp(run_id);

if ~exist('type_flag','var'), type_flag='train'; end
disp(type_flag);

file_name='inner_rf_samples_dynamic_';
run_id_str=num2str(run_id);
output_path='C:\Users\sinha_s\Desktop\dlmfg_package\dlmfg\active_learning\sample_input\inner_rf_assembly';

ai_path=append(output_path,'\',file_name,type_flag,'_',run_id_str,'.csv');

% install source codes
fprintf('     Installing src code...\n');
fem_src=fullfile(cd,'../FEM (src)');
if ~exist(fem_src,'dir')
    error('Installation of VRM (error): failed to locate "FEM(src)" folder!')
end
addpath(genpath(fem_src));
%
fao_src=fullfile(cd,'../FAO (src)');
if ~exist(fao_src,'dir')
    error('Installation of VRM (error): failed to locate "FAO(src)" folder!')
end
addpath(genpath(fao_src));
%
% install demo folder
fprintf('     Installing Demo pack...\n');
demos_src=fullfile(cd,'../Demos');
if ~exist(demos_src,'dir')
    error('Installation of VRM (error): failed to locate "Demos" folder!')
end
addpath(genpath(demos_src));
%
% install GUI
fprintf('     Installing GUI...\n');
gui_src=fullfile(cd,'../GUI');
if ~exist(gui_src,'dir')
    error('Installation of VRM (error): failed to locate "GUI" folder!')
end
addpath(genpath(gui_src));

fprintf('Installation completed!\n');
%
%% Define input files
%
%   Mesh file
%
mhfile{1}=fullfile(findfile(cd, 'door_inner_multi_station_4.inp'),'door_inner_multi_station_4.inp');
mhfile{2}=fullfile(findfile(cd, 'hinge_multi_station_4.inp'),'hinge_multi_station_4.inp');
% %
% PinHole file
%
pinholefile=fullfile(findfile(cd, 'pinhole_multi_station_4.txt'),'pinhole_multi_station_4.txt');
%
% PinSlot file
%
pinslotfile=fullfile(findfile(cd, 'pinslot_multi_station_4.txt'),'pinslot_multi_station_4.txt');
%
%
% NCBlock file
%
ncblockfile=fullfile(findfile(cd, 'NCBlock_multi_station_4.txt'),'NCBlock_multi_station_4.txt');
%
%   ClampS file
%
clampsfile=fullfile(findfile(cd, 'clampS_multi_station_4.txt'),'clampS_multi_station_4.txt');
%
%   ClampM file
%
clampmfile=fullfile(findfile(cd, 'clampM_multi_station_4.txt'),'clampM_multi_station_4.txt');
%
%   Stitch file
%
stitchfile=fullfile(findfile(cd, 'stitch_multi_station_4.txt'),'stitch_multi_station_4.txt');
%
%   Contact file
%
contactfile=fullfile(findfile(cd, 'contact_multi_station_4.txt'),'contact_multi_station_4.txt');
%

filenameX=fullfile(findfile(cd, ai_path),ai_path);
X_from_AI = importdata(filenameX); % [deg,mm,mm,mm,mm,mm]
% %
% % Convert the Angle (parameter(3)) in radians
X_from_AI(:,1)=X_from_AI(:,1)*pi/180;
%% Define material properties
% * Thickness
% * Young's Modulus
% * Poisson's ratio
Th=[1.5 2];
E=[69e3 69e3];
nu=[0.3 0.3];
%
%% Define stations
idStation=1; % place
stationData(idStation)=initStation();
stationData(idStation).Part=[1 2];
stationData(idStation).PinHole=[1 2];
stationData(idStation).PinSlot=[1 2];
stationData(idStation).Type{1}=2;
%
idStation=2; % clamp
stationData(idStation)=initStation();
stationData(idStation).Part=[1 2];
stationData(idStation).PinHole=[1 2];
stationData(idStation).PinSlot=[1 2];
stationData(idStation).ClampS=[1 3]; 
stationData(idStation).ClampM=[1:3];
stationData(idStation).Stitch=[];
stationData(idStation).NcBlock=[1:3];
stationData(idStation).Contact=[1];
stationData(idStation).Type{1}=3;  
%
idStation=3; % fasten
stationData(idStation)=initStation();
stationData(idStation).Part=[1 2];
stationData(idStation).PinHole=[1 2];
stationData(idStation).PinSlot=[1 2];
stationData(idStation).ClampS=[1 3]; 
stationData(idStation).ClampM=[1:7];
stationData(idStation).Stitch=[];
stationData(idStation).NcBlock=[1:3];
stationData(idStation).Contact=[1];
stationData(idStation).Type{1}=3; 
%
idStation=4; % release
stationData(idStation)=initStation();
stationData(idStation).Part=[1 2];
stationData(idStation).PinHole=[1 2];
stationData(idStation).PinSlot=[1 2];
stationData(idStation).ClampS=[1 3]; 
stationData(idStation).ClampM=[];
stationData(idStation).Stitch=[1:4];
stationData(idStation).NcBlock=[1:3];
stationData(idStation).Contact=[1];
stationData(idStation).Type{1}=4; 
%
%% Establish connection with database
nStation=length(stationData);
for i=1:nStation
    output_x{i}=sprintf('%sinner_rf_dynamic_dev_x_%g_%s_%g.csv','C:\Users\sinha_s\Desktop\dlmfg_package\dlmfg\datasets\inner_rf_assembly\',i,type_flag,run_id);
    output_y{i}=sprintf('%sinner_rf_dynamic_dev_y_%g_%s_%g.csv','C:\Users\sinha_s\Desktop\dlmfg_package\dlmfg\datasets\inner_rf_assembly\',i,type_flag,run_id);
    output_z{i}=sprintf('%sinner_rf_dynamic_dev_z_%g_%s_%g.csv','C:\Users\sinha_s\Desktop\dlmfg_package\dlmfg\datasets\inner_rf_assembly\',i,type_flag,run_id);
end

%
%% Initialize model
model.database=initDatabase();
%
% Solver settings
model.database.Assembly.Solver.UseParallel=true; % enable/disable parellel pool
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
    %
    model.database.Input.Part(i).Graphic.ShowEdge=true;
    model.database.Input.Part(i).Graphic.FaceAlpha=0.0;
    model.database.Input.Part(i).Graphic.Color=rand(1,3);
    model.database.Input.Part(i).Graphic.ShowNormal=false;
    model.database.Input.Part(i).Placement.ShowFrame=true;
end
model.database.Input.Part(2).Parametrisation.UCS=1;
model.database.Input.Part(2).Parametrisation.Type(3)=1; % Model training paramater[1] - rotation around pin
model.database.Input.Part(2).Parametrisation.Type(4)=1; % Model training paramater[2] - translation along pin
model.database.Input.Part(2).Parametrisation.Type(5)=1; % Model training paramater[3] - translation along pin
%
%% Define locator layout
%
% Hole
model.database=modelImportInput(model.database, pinholefile, 'Hole');
for i=1:length(model.database.Input.PinLayout.Hole)
    model.database.Input.PinLayout.Hole(i).Geometry.Shape.Parameter.Diameter=20;
    model.database.Input.PinLayout.Hole(i).Parametrisation.Geometry.ShowFrame=true;
    model.database.Input.PinLayout.Hole(i).SearchDist=[10 80];
    model.database.Input.PinLayout.Hole(i).TangentType{1}=1;
    model.database.Input.PinLayout.Hole(i).Nt=[1 0 0];
    model.database.Input.PinLayout.Hole(i).NtReset=[1 0 0];
end
%
% Slot
model.database=modelImportInput(model.database, pinslotfile, 'Slot');
for i=1:length(model.database.Input.PinLayout.Slot)
    model.database.Input.PinLayout.Slot(i).Geometry.Shape.Parameter.Diameter=20;
    model.database.Input.PinLayout.Slot(i).Geometry.Shape.Parameter.Length=10;
    model.database.Input.PinLayout.Slot(i).Parametrisation.Geometry.ShowFrame=true;
    model.database.Input.PinLayout.Slot(i).SearchDist=[10 10];
end
%
% NCBlock
model.database=modelImportInput(model.database, ncblockfile, 'NcBlock');
for i=1:length(model.database.Input.Locator.NcBlock)
    model.database.Input.Locator.NcBlock(i).Geometry.Shape.Type{1}=2; 
    model.database.Input.Locator.NcBlock(i).Geometry.Shape.Parameter.A=20;
    model.database.Input.Locator.NcBlock(i).Geometry.Shape.Parameter.B=20;
    model.database.Input.Locator.NcBlock(i).Geometry.Shape.Parameter.D=20;
    model.database.Input.Locator.NcBlock(i).Geometry.Shape.Parameter.L=50;
    model.database.Input.Locator.NcBlock(i).Geometry.Type{1}=2;
    model.database.Input.Locator.NcBlock(i).Parametrisation.Geometry.ShowFrame=true;
    model.database.Input.Locator.NcBlock(i).NormalType{1}=1;
    model.database.Input.Locator.NcBlock(i).Nm=[0 1 0];
    model.database.Input.Locator.NcBlock(i).NmReset=[0 1 0];
    model.database.Input.Locator.NcBlock(i).SearchDist=[100 100];
end
%
% ClampS
model.database=modelImportInput(model.database, clampsfile, 'ClampS');
for i=1:length(model.database.Input.Locator.ClampS)
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Type{1}=2; 
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Parameter.A=20;
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Parameter.B=20;
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Parameter.D=20;
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Parameter.L=50;
    model.database.Input.Locator.ClampS(i).Geometry.Type{1}=2;
    model.database.Input.Locator.ClampS(i).Parametrisation.Geometry.ShowFrame=true;
    model.database.Input.Locator.ClampS(i).NormalType{1}=1;
    model.database.Input.Locator.ClampS(i).Nm=[0 1 0];
    model.database.Input.Locator.ClampS(i).NmReset=[0 1 0];
    model.database.Input.Locator.ClampS(i).SearchDist=[100 100];
end
%
% ClampM
model.database=modelImportInput(model.database, clampmfile, 'ClampM');
for i=1:length(model.database.Input.Locator.ClampM)
    if i<4 % clamp
        model.database.Input.Locator.ClampM(i).Geometry.Shape.Type{1}=2; 
        model.database.Input.Locator.ClampM(i).Parametrisation.Geometry.ShowFrame=true;
        model.database.Input.Locator.ClampM(i).Graphic.Color='m';
    else % welding gun
        model.database.Input.Locator.ClampM(i).Geometry.Shape.Type{1}=1; 
        model.database.Input.Locator.ClampM(i).Graphic.ShowNormal=true;
        model.database.Input.Locator.ClampM(i).Graphic.Color='g';       
    end
    model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.A=20;
    model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.B=20;
    model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.D=20;
    model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.L=50;
    model.database.Input.Locator.ClampM(i).Geometry.Type{1}=2;
    model.database.Input.Locator.ClampM(i).SearchDist=[100 10];
end
for i=1:3
    model.database.Input.Locator.ClampM(i).Parametrisation.Geometry.Type{1}{1}=4; % Model training paramater[4 to 6]
    model.database.Input.Locator.ClampM(i).NormalType{1}=1;
    model.database.Input.Locator.ClampM(i).Nm=[0 1 0];
    model.database.Input.Locator.ClampM(i).NmReset=[0 1 0];
end
%
%% Define stitches
model.database=modelImportInput(model.database, stitchfile, 'Stitch');
for i=1:length(model.database.Input.Stitch)
    model.database.Input.Stitch(i).Diameter=20;
    model.database.Input.Stitch(i).SearchDist=[20 20];
    model.database.Input.Stitch(i).Gap=2.0; % mm
end
%
%% Define contact pairs
model.database=modelImportInput(model.database, contactfile, 'Contact');
model.database.Input.Contact(1).SearchDist=[20 80]; % normal distance/sharp angle
%
%% Build reference model
model.database=modelBuildPart(model.database,[1 1 1 1 1]); 
model.database=modelBuildInput(model.database, [1 0 0]);
%
%% Get list of parameters
model.database=modelGetParameters(model.database);
if isempty(model.database.Assembly.Parameter)
    error('Failed to extract model parameters!')
end
%
%% Run sampling
model.database.Assembly.SamplingStrategy{1}=3;
model.database.Assembly.SamplingOptions.IdTable=1;
model.database=modelAddItem(model.database, 'Parameter');
model.database.Input.Parameter(1).X=X_from_AI;
% model.database.Assembly.SamplingStrategy{1}=1;
% model.database.Assembly.SamplingOptions.SampleSize=1;
model.database.Assembly.Group=[1 -2*pi/180 2*pi/180 1
                               2 -1 1 1
                               3 -1 1 1
                               4 -2 2 1
                               5 -2 2 1
                               6 -2 2 1]; % [group ID, min, max, resolution]
model.database.Assembly.Parameter(1:end,2)={1;2;3;4;5;6};
model.database=modelSampleParameters(model.database);
%
%% Run simulation
model.database.Assembly.Solver.Eps=0.1;
model.database.Assembly.Solver.MaxIter=200;
model.database.Assembly.Solver.LinearSolver{1}=2;
model.database.Assembly.Solver.PenaltyStiffness=1e9;
model.database.Assembly.Solver.UseSoftSpring=false;
model.database.Assembly.Solver.SoftSpring=0.1;
%
useparallel=model.database.Assembly.Solver.UseParallel;
%
[nSimulations, nvars]=size(model.database.Assembly.X.Value);
%

%
flagSimulation=false(nStation, nSimulations);
nnode=size(model.database.Model.Nominal.xMesh.Node.Coordinate,1);
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

%% Write outputs for AI 

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
        
        % write data on file for each station
%         save(output_x{stationID}, 'Dx','-ascii');
%         save(output_y{stationID}, 'Dy','-ascii');
%         save(output_z{stationID}, 'Dz','-ascii');  

        csvwrite(output_x{stationID},Dx);
        csvwrite(output_y{stationID},Dy);
        csvwrite(output_z{stationID},Dz);
end
%write inputs if generated within VRM
%csvwrite(input_X,Xpara);
%% Show results
if ~showResults
    return
end

figure('units','normalized','outerposition',[0 0 1 1],'renderer','opengl', 'color','w')
ax=axes;
set(ax,'clipping', 'off','visible','off');
axis equal, hold all, grid on, box on

lighting phong
light
material metal
camproj('perspective')
colorbar
view([200 30]) 
camzoom(2)

plot_sim_training(model, stationData, U, [], inputData,ax);
 