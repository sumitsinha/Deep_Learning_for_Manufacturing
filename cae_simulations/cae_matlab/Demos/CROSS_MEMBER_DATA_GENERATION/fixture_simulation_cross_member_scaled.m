%--
% This script run assembly simulation of cross member assembly.
% 3 Stations 13 Stages | 1 (Non-ideal) + 4 (PCFR) + 4 (PCFR) + 4 (PCFR)
% 148 Process Paramaters | 123 continous | 25 Binary | (259 Code Wise)
%---

clc
clear
close all

%% Define input training model parameters (Sampled From AI)

% KCC 1 - 6 Part Variation (17 Control Points constraint to 4 Global and 2 Local Patterns)
%[Part1_Global Part2_Global Part3_Global Part3_Local Part4_Global Part4_Local]

X_from_AI_PV=[0.5 -0.7 1 0.2 -0.3 -1.1];

% KCC 7 - 9 Station 1 Part 1 Postitoning
%[z_rot x_dev y_dev]
X_from_AI_Pin_part1_station1=[2*pi/180 0 0];

% KCC 13 - 15 Station 3 Subassembly 1(Part 1(master)+ Part 2 (slave))Postitoning
%[z_rot x_dev y_dev]
X_from_AI_Pin_part1_station3=[-4*pi/180 0 0];

% KCC 10 - 12 Station 2 Part 3 Postitoning 
%[z_rot x_dev y_dev]
X_from_AI_Pin_part3_station2=[-2*pi/180 0 0];

% % KCC Station 1 Part 2 Postitoning inactive
% %[z_rot x_dev y_dev]
% X_from_AI_Pin_part2_station1=[0 0 0];

% KCC Station 2 Part 3 Postitoning inactive
%[z_rot x_dev y_dev]
% X_from_AI_Pin_part4_station2=[0 0 0];

% KCC Station 2 Subassembly 2(Part 3(master)+ Part 4(slave))Postitoning inactive
%[z_rot x_dev y_dev]
% X_from_AI_Pin_part4_station2=[0 0 0];

% KCC 16 - 42 | 9 (*3) ClampS across all stations 
%[x_dev y_dev z_dev]
X_from_AI_clampS=zeros([1,27]);
X_from_AI_clampS(25)=0.5; % X
X_from_AI_clampS(26)=0.7; % Y
X_from_AI_clampS(27)=0.9; % Z

% KCC 43 - 48 | 2 (*3) ClampM across all stations 
%[x_dev y_dev z_dev]
X_from_AI_clampM=zeros([1,6]);
X_from_AI_clampM(4)=1;
X_from_AI_clampM(5)=-1; % Y
X_from_AI_clampM(6)=1; % Z

% KCC 49 - 148 | 100 (*4) Joints (ClampM) across all stations 
%[x_dev y_dev z_dev on/off]
X_from_AI_joining=zeros([1,100]);
X_from_AI_joining(4:4:end)=1;
X_from_AI_joining(1)=0; % X
X_from_AI_joining(2)=0; % Y
X_from_AI_joining(3)=0; % Z
%X_from_AI_joining(4)=0; % % Flag

%stiches to follow same paramters as respective clampM
X_from_AI_joining_stiches=X_from_AI_joining;

% Combined KCC Array
X_from_AI = cat(2,X_from_AI_PV,X_from_AI_Pin_part1_station1,X_from_AI_Pin_part1_station3,X_from_AI_Pin_part3_station2,X_from_AI_joining_stiches,X_from_AI_clampS,X_from_AI_clampM,X_from_AI_joining);

%% AI IMPORT EXPOXT PARAMETERS
% Import AI generated file
ai_path='C:\Users\sinha_s\Desktop\dlmfg_package\dlmfg\active_learning\sample_input\cross_member_assembly\cross_member_samples_datagen_hybrid_5.csv';
filenameX=fullfile(findfile(cd, ai_path),ai_path);
X_from_AI = importdata(filenameX);

%Convert Rotation into Radians when importing from AI
X_from_AI(:,7)=X_from_AI(:,7)*pi/180;
X_from_AI(:,10)=X_from_AI(:,10)*pi/180;
X_from_AI(:,13)=X_from_AI(:,13)*pi/180;

%Convert into 
X_file_AI_pv=X_from_AI(:,1:6);
X_file_AI_positioning=X_from_AI(:,7:15);
X_file_AI_clampS=X_from_AI(:,16:42);
X_file_AI_clampM=X_from_AI(:,43:48);
X_file_AI_joining=X_from_AI(:,49:148);
X_file_AI_joining_stiches=X_file_AI_joining;

% Combined KCC Array
X_from_AI = cat(2,X_file_AI_pv,X_file_AI_positioning,X_file_AI_joining_stiches,X_file_AI_clampS,X_file_AI_clampM,X_file_AI_joining);

%%
%Change Path based on System
pathdestsave='C:\Users\sinha_s\Desktop\VRM (S 29 9 2020)\Demos\Cross member assembly\export_files';

%% Call Function to Import and Define input files
[mhfile,mmpfile,mmlfile,pinholefile,pinslotfile,stitchfile,Ncblockfile,clampSfile,clampMfile,contactfile]=cross_member_input_files ();

%% Initialize model
model=initModel();
% Solver settings
model.database.Assembly.Solver.UseParallel=true; % enable/disable parellel pool
showResults=false; % show/hide visualisation of results

%% Call Function to Define Station Structure
stationData=cross_member_station_structure();
%% Call Function to Define parts and properties
model=cross_member_part_property(model,mhfile);
%% Call Function to Import Part Variation Setup
model=cross_member_part_variation (model, mmpfile, mmlfile);
%% Call Function to setup part placement
model=cross_member_placement(model,pinholefile,pinslotfile);
%% Call Function to Define stitches
model=cross_member_stiches (model,stitchfile);
%% Call Function to Define clampM
model=cross_member_clampM (model,clampMfile);
%% Call Function to Define clampS
model=cross_member_clampS (model,clampSfile);
%% Call Function to Define NC Block
model=cross_member_NCblock (model,Ncblockfile);
%% Call Function to Define Part to Part Contact
model=cross_member_part_contact(model,contactfile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define parameter for part placement
model=parameter_cross_member_placement (model);
%% Define parameter for part clamping
model=parameter_cross_member_clamping (model);
%% Define parameter for part joining
model=parameter_cross_member_joining (model);

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

% create group of parameters
%- non-ideal parameters - part 1 
for i=1:4
    model.database.Assembly.Parameter{i,2}=1;
end
model.database.Assembly.X.Value(1:4)=[1 1 1 1]; % initial value
%
%- non-ideal parameters - part 2
for i=5:8
    model.database.Assembly.Parameter{i,2}=2; 
end
model.database.Assembly.X.Value(5:8)=[1 -1 1 -1]; % initial value
%
%- non-ideal parameters - part 3 (global)
for i=9:11
    model.database.Assembly.Parameter{i,2}=3; 
end
model.database.Assembly.X.Value(9:11)=[1 -1 1]; % initial value
%
%- non-ideal parameters - part 3 (local)
model.database.Assembly.Parameter{12,2}=4; 
model.database.Assembly.X.Value(12)=1; % initial value
%--
%- non-ideal parameters - part 4 (global)
for i=13:16
    model.database.Assembly.Parameter{i,2}=5; 
end
model.database.Assembly.X.Value(13:16)=[-1 -1 1 1]; % initial value
%
%- non-ideal parameters - part 4 (local)
model.database.Assembly.Parameter{17,2}=6; 
model.database.Assembly.X.Value(17)=1; % initial value
%--

%% Assign initial values for all paramaters apart from Part Variation
% initial values set to 1 which is multiplicative to the process parameter
% from AI
%
param_id=7;
iter_max=npara+11;
%
for i=18:iter_max
    model.database.Assembly.Parameter{i,2}=param_id;
    model.database.Assembly.X.Value(i)=1;
    param_id=param_id+1;
end

nSimulations=size(X_from_AI,1);
model.database.Assembly.X.Value=repmat(model.database.Assembly.X.Value,nSimulations,1);
%
%% Sample Parameters
model.database=modelSampleParameters(model.database);
%
%% Run simulation
model.database.Assembly.Solver.Eps=0.5;
model.database.Assembly.Solver.MaxIter=100;
model.database.Assembly.Solver.LinearSolver{1}=2;
model.database.Assembly.Solver.PenaltyStiffness=1e9;
model.database.Assembly.Solver.UseSoftSpring=true;
model.database.Assembly.Solver.SoftSpring=0.2;
%
useparallel=model.database.Assembly.Solver.UseParallel;
%
[nSimulations, nvars]=size(model.database.Assembly.X.Value);
nStation=length(stationData);
U=cell(1, nSimulations);
GAP=cell(1, nSimulations);
inputData=cell(1, nSimulations);
FLAG=true(nSimulations, nStation);

if useparallel % USE PARALLEL MODE
    parfor paraID=1:nSimulations
        [U{paraID},...
         GAP{paraID},...
         FLAG(paraID,:),...
         inputData{paraID}]=modelSolve(model.database, stationData, paraID);  %#ok<PFBNS>
    end
else % USE SEQUENTIAL MODE
    for paraID=1:nSimulations
        [U{paraID},...
         GAP{paraID},...
         FLAG(paraID,:),...
         inputData{paraID}]=modelSolve(model.database, stationData, paraID); 
    end
end
%
% Save back
model.Session.Station=stationData;
model.Session.Input=inputData;
model.Session.U=U;
model.Session.Gap=GAP;
model.Session.Status=FLAG;

%% Updated File Saving
[file_model,...
        file_input,...
        file_station,...
        file_U,...
        file_gap,...
        file_flag,...
        file_x, file_y, file_z,...
        file_Para,...
        file_statusPara,...
        file_AIPara]= modelExportDatasetFormatFiles(pathdestsave, nStation);
%
opt=[1 0 0 0 0 0 0 1 1 1 1]; % export whole dataset from ".Session"
modelExportDataset(file_model,...
                    file_input,...
                    file_station,...
                    file_U,...
                    file_gap,...
                    file_flag,...
                    file_x, file_y, file_z,...
                    file_Para,...
                    file_statusPara,...
                    file_AIPara,...
                    model, opt);                  

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
camzoom(1.3)
%
model.Figure=fig;
model.Axes3D.Axes=ax;
model.Axes3D.Options.ShowAxes=true;
model.Axes3D.Options.LengthAxis=40;
model.Axes3D.Options.SymbolSize=10;

paraID=1;
stationID=1;
contourVar=1;
dataRange=[-1 1];
deformedScale=1;
deformedFlag=true;
animationDelay=0.1;
animateFlag=false;
opt=1;
nominalFlag=[true true false false];
contourFlag=[true true true true];

render_contour(model,...
                paraID,...
                stationID,...
                contourVar,...
                dataRange,...
                deformedScale,...
                deformedFlag,...
                animationDelay,...
                animateFlag,...
                opt,...
                nominalFlag,...
                contourFlag)
            
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
              'callback',{@standard_figure_render, model, paraID, contourVar, dataRange, deformedScale, deformedFlag,...
                          nominalFlag,contourFlag})
end
