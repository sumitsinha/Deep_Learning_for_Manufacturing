function run_simulation_gui(source, event, h)

import javax.swing.*;

data=guidata(h);

if ~data.Session.Flag
    st=get(data.logPanel,'string');
    st{end+1}='Error: failed to load model from current session!';
    set(data.logPanel, 'string',st);
    return
end
%
font_size=data.Options.FontSize;
%
% delete previous objs
delete(get(data.setPanel,'children'))
%
% Count no. of parameters
nParameters=size(data.Simulation.Parameters,2);
parameterList=data.database.Assembly.Parameter;
if nParameters==0
    st=get(data.logPanel,'string');
    st{end+1}='Error - no model has been identified!';
    set(data.logPanel, 'string',st);
    return
end
st=get(data.logPanel,'string');
st{end+1}=sprintf('Message - no. of parameters: %g',nParameters);
set(data.logPanel, 'string',st); 
%
% model settings
panv=uipanel('Parent',data.setPanel,'Units', 'normalized','Position',[0.02,0.1,0.95,0.9],...
            'Title','Model settings',...
            'FontSize',font_size);
%   
% Create table with paramaters
d=cell(nParameters,5);
X=data.Simulation.Parameters; X=X(1,:);
for i=1:1:nParameters
    parameterMode=parameterList{i,1};
    parameterDir=parameterList{i,4};
    parameterType=parameterList{i,7};
    
    d{i,5}='Continuous';
    
    if parameterMode==1
        if parameterDir<=3
            Xdefault=[-data.Simulation.Options.MaxRotation X(i)*180/pi data.Simulation.Options.MaxRotation];
            d{i,1}=sprintf('X(%g) - [deg]',i);
        else
            Xdefault=[-data.Simulation.Options.MaxTranslation X(i) data.Simulation.Options.MaxTranslation];
            d{i,1}=sprintf('X(%g) - [mm]',i);
        end
    elseif parameterMode==0
            Xdefault=[-data.Simulation.Options.MaxTranslation X(i) data.Simulation.Options.MaxTranslation];
            d{i,1}=sprintf('X(%g) - [mm]',i);
    elseif parameterMode==2 
        if strcmp(parameterType,'ON/OFF')
            Xdefault=[0 X(i) 1];
            d{i,1}=sprintf('X(%g) - [--]',i);
            %--
            d{i,5}='Categorical';
            %--
        else
            Xdefault=[-data.Simulation.Options.MaxTranslation X(i) data.Simulation.Options.MaxTranslation];
            d{i,1}=sprintf('X(%g) - [mm]',i);
        end
    end
    for j=2:4
        d{i,j}=Xdefault(j-1);
    end
end
%
cnames={'Parameter','Value (min)', 'Value (actual)','Value (max)', 'Type'};
formcolum={'Char','Numeric', 'Numeric','Numeric' 'Char'};
uitable('Data',d, 'ColumnName',cnames,'parent',panv,...
        'ColumnEditable', [false false true false false],'SelectionHighlight','on',...
        'tag','table', 'FontSize',font_size,...
        'Units', 'normalized','Position',[0,0,1,1],...
        'ColumnFormat', formcolum,...
        'CellSelectionCallback',{@cell_selected_table, h});
%--        
% button settings
b = uiextras.HButtonBox('Parent', data.setPanel, 'Units', 'normalized','Position',[0,0.0,1,0.1]);

uicontrol( 'Parent', b, 'String', 'Manual Drag','Callback',{@play_menu, h}, 'FontSize',font_size)
uicontrol( 'Parent', b, 'String', 'Reset','Callback',{@reset_menu, h}, 'FontSize',font_size)
uicontrol( 'Parent', b, 'String', 'Solve','Callback',{@solve_menu, h}, 'FontSize',font_size)
uicontrol( 'Parent', b, 'String', 'Cancel','Callback', {@close_menu, h}, 'FontSize',font_size)
set(b,'ButtonSize',[100 50])

%--
function play_menu(source, event, h)

data=guidata(h);

% Plot paramaters
parameterList=data.database.Assembly.Parameter;
%
% Reset graphics
reset_rendering(data, 2); reset_rendering(data, 4);
% Plot
%
motionData.Enable=true;
motionData.Fig=data.Figure(1);
motionData.Axes=data.Axes3D.Axes;
motionData.Target=data.Axes3D.Options.Tag.TempObject;
motionData.SpeedCameraMotion=data.Axes3D.Options.SpeedCameraMotion;
%
if isempty(data.Table.Selection)
    st=get(data.logPanel,'string');
    st{end+1}='Error - please select a parameter from the table!';
    set(data.logPanel, 'string',st);
    return
end
paraID=data.Table.Selection; paraID=paraID(1);
parameterMode=parameterList{paraID,1};
%
logData.Panel.Object=findobj(h,'tag','table');
logData.Panel.ID=paraID;
%
if parameterMode==1
    partIDi=parameterList{paraID,3};
    paraTypei=parameterList{paraID,4};
    ucsTypei=parameterList{paraID,6};
    motionData.Direction=paraTypei;
    if paraTypei<=3
        motionData.MaxRotation=data.Simulation.Options.MaxRotation;
    else
        motionData.MaxTranslation=data.Simulation.Options.MaxTranslation;
    end
    %
    nStation=length(data.Session.Station);
    for stationID=1:nStation
        data.database=modelAssignUCSLocatorPlacement(data.database, data.Session.Station, stationID);
    end
    %
    if ucsTypei==0
        motionData.Rf=eye(3,3); 
        motionData.Pf=[0 0 0];
    elseif ucsTypei==1
        motionData.Rf=data.database.Input.Part(partIDi).Placement.UCS(1:3,1:3); 
        motionData.Pf=data.database.Input.Part(partIDi).Placement.UCS(1:3,4)'; 
    end
    %
    logData.motionData=motionData;
    %
    data.database.Model.Nominal.Post.Options.ParentAxes=data.Axes3D.Axes;
    data.database.Model.Nominal.Post.Options.ShowAxes=data.Axes3D.Options.ShowAxes;
    data.database.Model.Nominal.Post.Options.LengthAxis=data.Axes3D.Options.LengthAxis;
    tcol=data.database.Input.Part(partIDi).Graphic.Color;
    fcol=data.database.Input.Part(partIDi).Graphic.FaceAlpha;
    sframe=data.database.Input.Part(partIDi).Placement.ShowFrame;
    %
    data.database.Input.Part(partIDi).Graphic.Color='r';
    data.database.Input.Part(partIDi).Graphic.FaceAlpha=1;
    data.database.Input.Part(partIDi).Placement.ShowFrame=true;
    %
    modelPlotDataGeomSingle(data.database, 'Part', partIDi, [], data.Axes3D.Options.Tag.TempObject, logData)
    data.database.Input.Part(partIDi).Graphic.Color=tcol;
    data.database.Input.Part(partIDi).Graphic.FaceAlpha=fcol;
    data.database.Input.Part(partIDi).Placement.ShowFrame=sframe;
    
elseif parameterMode==2
    fieldi=parameterList{paraID,3};
    fieldiID=parameterList{paraID,4};
    paraidi=parameterList{paraID,5};
    paratypei=parameterList{paraID,7};
    %
    if strcmp(paratypei,'ON/OFF')
        logData.motionData=[];
    else
        motionData.MaxTranslation=data.Simulation.Options.MaxTranslation;
        motionData.Direction=parameterList{paraID,6}-1+3;
        logData.motionData=motionData;
    end
    plotDataInputSingle(data, fieldi, fieldiID, paraidi, logData, data.Axes3D.Options.Tag.TempObject);
    
elseif parameterMode==0
    idparti=parameterList{paraID,3};
    pointidi=parameterList{paraID,4};
    %
    motionData.MaxTranslation=data.Simulation.Options.MaxTranslation;
    motionData.Direction=6; % local normal vector
    %

    logData.motionData=motionData;
    morphPlotDomainSingle(data, idparti, pointidi, logData, data.Axes3D.Options.Tag.TempObject);
end

%--
function reset_menu(source, event, h)

% Reset graphics
data=guidata(h);

reset_rendering(data, 2); reset_rendering(data, 4);

% reset table
obj=findobj(h,'tag','table');
d=get(obj,'data');

n=size(d,1);
for i=1:n
    d{i,3}=0;
end
set(obj,'data',d);
%--

function parameterData=read_data(h)

data=guidata(h);

% Parameter data
obj=findobj(h,'tag','table');
parameterDatac=get(obj,'data');
nParameters=size(parameterDatac,1);
parameterData=zeros(1,nParameters);
parameterList=data.database.Assembly.Parameter;
%
valDef=0;
for i=1:nParameters
    val=parameterDatac{i,3};
    checkcond.b(1)=parameterDatac{i,4};
    checkcond.Type{1}='<=';
    checkcond.b(2)=parameterDatac{i,2};
    checkcond.Type{2}='>=';
    [pX, flag]=check_condition_double(val, valDef, checkcond);
    %
    if ~flag
        st=get(data.logPanel,'string');
        st{end+1}=sprintf('Warning @X(%g): selected parameters outside limits. Using default value',i);
        set(data.logPanel, 'string',st);
    end
    %
    parameterData(i)=pX;
    if parameterList{i,1}==1
        if parameterList{i,4}<=3
            parameterData(i)=pX*pi/180;
        end
    end      
end

function solve_menu(source, event, h)

data=guidata(h);

% STEP 1 - Read parameters
parameterData=read_data(h);

% STEP 2 - Sample parameters
data=solve_sample_parameters(data, parameterData);

% STEP 3 - Solve model
st=get(data.logPanel,'string');
st{end+1}='Message - simulation started, please wait...!';
set(data.logPanel, 'string',st);
pause(0.5);

stationData=data.Session.Station;
[U{1}, ~, ~, inputData{1}]=modelSolve(data.database, stationData, 1);
st=get(data.logPanel,'string');
st{end+1}='Message: simulation completed!';
set(data.logPanel, 'string',st);

% Save back
data.Simulation.U=U;
data.Simulation.Input=inputData;
data.Simulation.Flag=true;
guidata(h, data);
    
%--
function data=solve_sample_parameters(data, parameterData)

% STEP 0 - sample parameters
data.database.Assembly.SamplingStrategy{1}=3;
data.database.Assembly.SamplingOptions.IdTable=1; % Parameter table
%--
data.database=modelAddItem(data.database, 'Parameter');
data.database.Input.Parameter(1).X=parameterData;
nParameters=size(data.database.Assembly.Parameter,1);
for i=1:nParameters
    data.database.Assembly.Parameter{i,2}=i;
end
%--
data.database.Assembly.Group=zeros(nParameters, 4); data.database.Assembly.Group(1:end,1)=1:nParameters; 
%--
data.database=modelSampleParameters(data.database);

