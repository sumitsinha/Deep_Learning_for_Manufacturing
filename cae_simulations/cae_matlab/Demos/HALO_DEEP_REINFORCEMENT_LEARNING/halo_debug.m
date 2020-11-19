% This script validate on single component the integration of VRM and AI
%
% INPUTs from AI:
    % X_from_AI (parameters matrix); [nSimulations, nParameters]
    % ParameterType=[] % TBD
% OUTPUTs to AI:
    % Dx (deviation X): [nSimulations, nNode+1] - last column is the "flag"
    % Dy (deviation Y): [nSimulations, nNode+1] - last column is the "flag"
    % Dz (deviation Z): [nSimulations, nNode+1] - last column is the "flag"
    % U (full deviation field): [nDoF, nSimulations]
    % flag (boolean): [1, nSimulations]
    %
        % nSimulations: no. of sampled points for AI training
        % nParameters: no. of training paramaters (KCCs)
        % nNode: no. of KPIs
        % nDoF: no. of total DoFs in the VRM model
%
%--
clc
clear
close all
%
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
E=[69e6];
nu=[0.3];
%
%% Define parameters for part placement
plcParameter.Group=[1 -2 2 1
                    2 -2 2 1
                    3 -1*pi/180 1*pi/180 1]; % [group ID, min, max, res] (not need when importing from "Parameter Table")
plcParameter.Parameter=[1 1 4 1
                        2 1 5 1
                        3 1 3 1]; % [groupID, partID, parameter type, reference system]
%
%% Establish connection with database
% filenameX=fullfile(findfile(cd, 'initial_samples_debug.csv'),'initial_samples_debug.csv');
% X_from_AI = importdata(filenameX); % [mm, mm, deg]
% %
% % Convert the Angle (parameter(3)) in radians
% X_from_AI(:,3)=X_from_AI(:,3)*pi/180;
% %
% %subsetting for debugging
% X_from_AI=X_from_AI(1000:1100,:);

% output_x=['C:\Users\sinha_s\Desktop\dlmfg_package\VRM\Demos\Fixture simulation\Validation\', 'output_table_x.csv'];
% output_y=['C:\Users\sinha_s\Desktop\dlmfg_package\VRM\Demos\Fixture simulation\Validation\', 'output_table_y.csv'];
% output_z=['C:\Users\sinha_s\Desktop\dlmfg_package\VRM\Demos\Fixture simulation\Validation\', 'output_table_z.csv'];
% 
% input_X=['C:\Users\sinha_s\Desktop\dlmfg_package\VRM\Demos\Fixture simulation\Validation\', 'input_X.csv'];

%
%% Initialize model
model.database=initDatabase();
%
% Solver settings
model.database.Assembly.Solver.UseParallel=false; % enable/disable parellel pool
showResults=true; % show/hide visualisation of results
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
end
%
%% Define locator layout
%
% Hole
model.database=modelImportInput(model.database, pinholefile, 'Hole');
%
model.database.Input.PinLayout.Hole.Geometry.Shape.Parameter.Diameter=20;
model.database.Input.PinLayout.Hole.Parametrisation.Geometry.ShowFrame=true;
model.database.Input.PinLayout.Hole.TangentType{1}=1; % user
%
% Slot
model.database=modelImportInput(model.database, pinslotfile, 'Slot');
%
model.database.Input.PinLayout.Slot.Geometry.Shape.Parameter.Diameter=20;
model.database.Input.PinLayout.Slot.Geometry.Shape.Parameter.Length=20;
model.database.Input.PinLayout.Slot.Parametrisation.Geometry.ShowFrame=true;
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
    model.database.Input.Locator.ClampS(i).Graphic.FaceAlpha=0.3;
end
%
%% Build inputs
model.database=modelBuildPart(model.database,'import'); 
model.database=modelBuildInput(model.database);
%
%% Get part UCS (aligned to the hole)
R0h=model.database.Input.PinLayout.Hole.Parametrisation.Geometry.R{1};
P0h=model.database.Input.PinLayout.Hole.Pm;
Tucs=eye(4,4); Tucs(1:3,1:3)=R0h; Tucs(1:3,4)=P0h; 

model.database.Input.Part.Placement.UCS=Tucs; % part UCS aligned to the hole
%
%% Compute and assign parameters for placement
model.database.Assembly.SamplingStrategy{1}=1; % user 
setSample=[];
        % model.database=modelAddItem(model.database, 'Parameter');
        % model.database.Input.Parameter(1).X=X_from_AI;
        % setSample.IdTable=1; % Parameter table    
[Xpara, flagpara]=modelSamplePlacement(model.database, plcParameter.Parameter, plcParameter.Group, setSample);
if ~flagpara
    error('Failed to calculate parameter space - please check inputs!')
end
%
%% Get part definitions
[pdata, flag]=getPartDescription(model.database);
if ~flag
    error('Failed to calculate part descriptors - no part is currently available!')
end
%
%% Run simulation

%- Get dimensions
nnode=size(model.database.Model.Nominal.xMesh.Node.Coordinate,1);
[nSimulations,nvars]=size(Xpara);
nNoFs=model.database.Model.Nominal.Sol.nDoF;
%
% Prealloate output variables
Dx=zeros(nSimulations, nnode+1); % deviation X
Dy=zeros(nSimulations, nnode+1); % deviation Y
Dz=zeros(nSimulations, nnode+1); % deviation Z
U=zeros(nNoFs, nSimulations); % full deviations fields (u, v, w, alfa, beta, gamma)
%--
useparallel=model.database.Assembly.Solver.UseParallel;
if useparallel % USE PARALLEL MODE
    parfor paraID=1:nSimulations
        [Dxp,...
         Dyp,...
         Dzp,...
         Up,...
         flagSimulationsp]=run_local_sim(model.database, Xpara, plcParameter, pdata, paraID); %#ok<PFBNS>
         %
         if flagSimulationsp % solved
             Dx(paraID, :)=[Dxp', 1];
             Dy(paraID, :)=[Dyp', 1];
             Dz(paraID, :)=[Dzp', 1];
             U(:, paraID)=Up;
         end
    end
else % USE SEQUENTIAL MODE
    for paraID=1:nSimulations
         [Dxp,...
         Dyp,...
         Dzp,...
         Up,...
         flagSimulationsp]=run_local_sim(model.database, Xpara, plcParameter, pdata, paraID);
         %
         if flagSimulationsp % solved
             Dx(paraID, :)=[Dxp', 1];
             Dy(paraID, :)=[Dyp', 1];
             Dz(paraID, :)=[Dzp', 1];
             U(:, paraID)=Up;
         end
    end
end
%
% %% Save back to database
% save(output_x, 'Dx','-ascii');
% save(output_y, 'Dy','-ascii');
% save(output_z, 'Dz','-ascii');
% 
% save(input_X, 'Xpara','-ascii');
% 
% %%
% save(output_x, 'Dx','-ascii');
% save(output_y, 'Dy','-ascii');
% save(output_z, 'Dz','-ascii');
% 
% save(input_X, 'Xpara','-ascii');
% %%
% Xpara(:,3)=Xpara(:,3)*180/pi;
% csvwrite(output_x,Dx);
% csvwrite(output_y,Dy);
% csvwrite(output_z,Dz);
% csvwrite(input_X,Xpara);
%%
%% Show results
if ~showResults
    return
end
%
figure('units','normalized','outerposition',[0 0 1 1],'renderer','opengl', 'color','w')
ax=axes;
set(ax,'clipping', 'off','visible','off');
axis equal, hold all, grid on, box on

lighting phong
light
material metal
camproj('perspective')
colorbar
view(3)
camzoom(1.6)
%
% Init plotting options
model.Axes3D=initAxesPlot(ax);
model.Axes3D.Options.LengthAxis=100;
model.Axes3D.Options.SymbolSize=10;
model.Axes3D.Options.ShowAxes=false;
% Plot model geometry
model.database=getPlacementMatrix(model.database, zeros(size(Xpara,2),1), 1, plcParameter);
model.database=modelBuildPart(model.database,'refresh'); 
model.database=modelBuildInput(model.database,'features');
model.database.Input.Part.Graphic.FaceAlpha=0.0;
modelPlotDataGeom(model, 'Part');
% Plot assembly toolings
plotDataInput(model, 1);

% plot results
for paraID=1:nSimulations
    
    % Update placement
    model.database=getPlacementMatrix(model.database, Xpara, paraID, plcParameter);
    %
    % Build reference model
    model.database=modelBuildPart(model.database,'refresh'); 
    model.database=modelBuildInput(model.database,'features');
    
        %     model.database.Input.Part.Graphic.FaceAlpha=1.0;
        %     modelPlotDataGeom(model, 'Part','temp');
    % Plot assembly toolings
    for i=1:length(model.database.Input.Locator.ClampS)
        model.database.Input.Locator.ClampS(i).Graphic.Color='m';
        model.database.Input.Locator.ClampS(i).Graphic.FaceAlpha=1;
    end
    plotDataInput(model, 1,'temp');
    
    %
    fem=model.database.Model.Nominal;
    fem.Post.Options.ParentAxes=ax;
    fem.Post.Options.ShowAxes=model.Axes3D.Options.ShowAxes;
    fem.Post.Contour.Deformed=true; % plot with deformation
    fem.Post.Contour.ScaleFactor=1; % scale factor
    fem.Post.Contour.MaxRange=2; % max color range
    fem.Post.Contour.MinRange=-2;  % min color range
    fem.Post.Contour.Resolution=1; % resolution of the contour plot
    fem.Post.Contour.MaxRangeCrop=inf; % max cropping limit
    fem.Post.Contour.MinRangeCrop=-inf; % min cropping limit
    fem.Post.Contour.ContactPair=1;
    fem.Post.Contour.ContourVariable='v'; % variable to plot
    
    fem.Sol.U=U(:, paraID);
    
    fem.Post.Contour.Domain=1; 
    contourPlot(fem,'temp')
        
    caxis([fem.Post.Contour.MinRange fem.Post.Contour.MaxRange])
    
    pause(0.1)
    if paraID==1
        xlim=get(gca,'xlim');
        ylim=get(gca,'ylim');
        zlim=get(gca,'zlim');
    end
    %   
    if paraID<nSimulations
        set(gca,'xlim',xlim)
        set(gca,'ylim',ylim)
        set(gca,'zlim',zlim)
        delete(findobj(gcf,'tag','temp'));
    end
end
 