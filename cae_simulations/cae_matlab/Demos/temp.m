%---


%--
clc
clear
close all
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
%
%   ClampM file
%
clampmfile=fullfile(findfile(cd, 'clampM_multi_station_4.txt'),'clampM_multi_station_4.txt');
%
%   Contact file
%
contactfile=fullfile(findfile(cd, 'contact_multi_station_4.txt'),'contact_multi_station_4.txt');
%
%% Define material properties
% * Thickness
% * Young's Modulus
% * Poisson's ratio
Th=[1.5 2];
E=[69e3 69e3];
nu=[0.3 0.3];
%
%% Define stations
stationData(1)=initStation();
stationData(1).Part=[1 2];
stationData(1).PinHole=[1 2];
stationData(1).PinSlot=[1 2];
stationData(1).Contact=[1];

% stationData(1).ClampS=[1:3];
% stationData(1).ClampM=[1:3];
% stationData(1).NcBlock=[1:3];
% stationData(1).Contact=[1];


stationData(1).Type{1}=2;
stationData(1).Parameter.Group=[1 -2 -1 1
                                2 -2 -2 1
                                3 -2.5*pi/180 -2.5*pi/180 1
                                4 -2*0 2*0 1
                                5 -2*0 2*0 1
                                6 -0.5*pi/180*0 0.5*pi/180*0 1]; % [group ID, min, max, res]
stationData(1).Parameter.Parameter=[3 2 3 1]; % [groupID, partID, parameter type, reference system]
% 
stationData(2)=initStation();
stationData(2).Part=[1 2];
stationData(2).PinHole=[1 2];
stationData(2).PinSlot=[1 2];
stationData(2).ClampS=[1 3]; % note: removed clampS[2]
stationData(2).ClampM=[1:7];
stationData(2).NcBlock=[1:3];
stationData(2).Contact=[1];
stationData(2).Type{1}=3;

stationData(3)=initStation();
stationData(3).Part=[1 2];
stationData(3).PinHole=[1 2];
stationData(3).PinSlot=[1 2];
stationData(3).ClampS=[1 3]; % note: removed clampS[2]
stationData(3).ClampM=[1:3];
stationData(3).NcBlock=[1:3];
stationData(3).Contact=[1];
stationData(3).Type{1}=3;

stationData(4)=initStation();
stationData(4).Part=[1 2];
stationData(4).PinHole=[1 2];
stationData(4).PinSlot=[1 2];
stationData(4).ClampS=[1 3]; % note: removed clampS[2]
stationData(4).ClampM=[4:7];
stationData(4).NcBlock=[1:3];
stationData(4).Contact=[1];
stationData(4).Type{1}=3;


% stationData(1).ClampS=[1:3];
% stationData(1).ClampM=[];
% stationData(1).NcBlock=[];


% stationData(1).Parameter.Parameter=[1 1 4 1
%                                     2 1 5 1
%                                     3 1 3 1
%                                     4 2 4 1
%                                     5 2 5 1
%                                     6 2 3 1]; 
                                
                                
%
%% Establish connection with database


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
    %
    model.database.Input.Part(i).Graphic.ShowEdge=true;
    model.database.Input.Part(i).Graphic.FaceAlpha=0.0;
    model.database.Input.Part(i).Graphic.Color=rand(1,3);
    model.database.Input.Part(i).Graphic.ShowNormal=false;
    model.database.Input.Part(i).Placement.ShowFrame=false;
end
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
    model.database.Input.Locator.ClampM(i).NormalType{1}=1;
    model.database.Input.Locator.ClampM(i).Nm=[0 1 0];
    model.database.Input.Locator.ClampM(i).NmReset=[0 1 0];
end
%
%% Define contact pairs
model.database=modelImportInput(model.database, contactfile, 'Contact');
model.database.Input.Contact(1).SearchDist=[20 80]; % normal distance/sharp angle
%
%% Build reference model
model.database=modelBuildPart(model.database,'import'); 
model.database=modelBuildInput(model.database);
%
%% --
model.database.Assembly.SamplingStrategy{1}=1;
setSample.SampleSize=5;
[Xpara, flagpara]=modelSamplePlacement(model.database, stationData(1).Parameter.Parameter, stationData(1).Parameter.Group, setSample);
if ~flagpara
    error('Failed to calculate parameter space - please check inputs!')
end
stationData(1).Parameter.X=Xpara;
%
%% Run simulation
model.database.Assembly.Solver.Eps=0.1;
model.database.Assembly.Solver.MaxIter=100;
model.database.Assembly.Solver.LinearSolver{1}=2;
model.database.Assembly.Solver.PenaltyStiffness=1e9;
model.database.Assembly.Solver.UseSoftSpring=false;
model.database.Assembly.Solver.SoftSpring=0.1;

[nSimulations, nvars]=size(Xpara);
U=cell(1, nSimulations);
GAP=cell(1, nSimulations);
inputData=cell(1, nSimulations);
for paraID=1:nSimulations
    stationData(1).Parameter.ID=paraID;
    [U{paraID}, GAP{paraID}, inputData{paraID}]=modelStationSolve(model.database, stationData);
end
%
%% Show results
figure('units','normalized','outerposition',[0 0 1 1],'renderer','opengl', 'color','w')
ax=axes;
set(ax,'clipping', 'off','visible','off');
axis equal, hold all, grid on, box on

lighting phong
light
material metal
camproj('perspective')
colorbar
view([-90 4]) 
camzoom(1)
%
% Init plotting options
model.Axes3D=initAxesPlot(ax);
model.Axes3D.Options.LengthAxis=100;
model.Axes3D.Options.SymbolSize=20;
model.Axes3D.Options.ShowAxes=true;

% Plot model geometry
model.database.Input.Part(1).Graphic.Show=true;
model.database.Input.Part(2).Graphic.Show=true;
modelPlotDataGeom(model, 'Part');
% Plot assembly toolings
% plotDataInput(model, 1);
% 
% pause(1)

fem=model.database.Model.Nominal;
inputData0=model.database.Input;

fem.Post.Options.ParentAxes=ax;
fem.Post.Options.ShowAxes=model.Axes3D.Options.ShowAxes;
fem.Post.Contour.Deformed=true; % plot with deformation
fem.Post.Contour.ScaleFactor=1; % scale factor
fem.Post.Contour.Resolution=1; % resolution of the contour plot
fem.Post.Contour.MaxRangeCrop=inf; % max cropping limit
fem.Post.Contour.MinRangeCrop=-inf; % min cropping limit
fem.Post.Contour.ContactPair=1;

% plot results
nStation=length(stationData);
c=1;
for paraID=1:nSimulations
    
    for stationID=1:nStation
        % plot inputs
        model.database.Input=inputData{paraID}{stationID};
        plotDataInput(model, 1, 'temp');
        model.database.Input=inputData0;
        %
        % plot contour plot
        fem.Sol.Gap=GAP{paraID}{stationID};
         if stationID==1
             fem.Sol.U=U{paraID}(:,1);
         elseif stationID==2
             fem.Sol.U=sum(U{paraID}(:,1:stationID),2);
         elseif stationID==3
             fem.Sol.U=sum(U{paraID}(:,[1:stationID]),2);
         end
            
        nDom=fem.Sol.nDom;
        for id=2%1:nDom
            if id==1
                fem.Post.Contour.MaxRange=0; % max color range
                fem.Post.Contour.MinRange=0;  % min color range
                fem.Post.Contour.ContourVariable='v'; % variable to plot  
            else
                fem.Post.Contour.MaxRange=2; % max color range
                fem.Post.Contour.MinRange=-2;  % min color range
%                 if stationID==2
                    fem.Post.Contour.ContourVariable='gap'; % variable to plot  
%                 else
%                     fem.Post.Contour.ContourVariable='v'; % variable to plot  
%                 end
            end
            fem.Post.Contour.Domain=id; 
            contourPlot(fem,'temp')
        end
        contourcmap('jet')

        caxis([fem.Post.Contour.MinRange fem.Post.Contour.MaxRange])
        
        title_text=sprintf('Sample generation[%g] @Station[%g]',paraID, stationID);
        title(title_text,'tag','temp')

        pause(1)
        if paraID==1
            xlim=get(gca,'xlim');
            ylim=get(gca,'ylim');
            zlim=get(gca,'zlim');
        end

        if c<nSimulations*nStation
            set(gca,'xlim',xlim)
            set(gca,'ylim',ylim)
            set(gca,'zlim',zlim)
            delete(findobj(gcf, 'tag','temp'));
        end
        
        c=c+1;
    end
end
 