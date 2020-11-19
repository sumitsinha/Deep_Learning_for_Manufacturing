% render model
function render_contour(data,...
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
         
% data: input model
% paraID: parameter ID - integer
% stationID: station ID - integer
% contourVar: variable to plot - [1 2 3] => {u, v, w}; [4] => gap
% dataRange: data range [min, max] - double
% deformedScale: deformation scale - double
% deformedFlag: enable/disable deformation plot - boolean
% animationDelay: animation delay - double
    % If animationDelay<0 => frames can be controlled from the console
% animateFlag: enable/disable animation of all parameters and all station - boolean
% opt
%     1: => render current session (data.Session)
%     2: => render simulation (data.Simulation) - only for GUI
% nominalFlag: show/hide nominal geometry (true if part must be plotted; false otherwise)
% contourFlag: show/hide contour geometry (true if part must be plotted; false otherwise)

if nargin<11
    nominalFlag=1;
end

%--------------
tag=data.Axes3D.Options.Tag.TempObject;
%--------------

% reset graphics
reset_rendering(data, 2);
%
% Set options for contour plot
data.database.Model.Nominal.Post.Options.ParentAxes=data.Axes3D.Axes;
data.database.Model.Nominal.Post.Options.ShowAxes=data.Axes3D.Options.ShowAxes;
data.database.Model.Nominal.Post.Contour.Deformed=deformedFlag; % plot with deformation
data.database.Model.Nominal.Post.Contour.ScaleFactor=deformedScale; % scale factor
data.database.Model.Nominal.Post.Contour.Resolution=1; % resolution of the contour plot
data.database.Model.Nominal.Post.Contour.MaxRangeCrop=inf; % max cropping limit
data.database.Model.Nominal.Post.Contour.MinRangeCrop=-inf; % min cropping limit
data.database.Model.Nominal.Post.Contour.ContactPair=1;
%
% Nominal input
inputData0=data.database.Input;
%
% Plot input and contours
if animateFlag
    nStation=length(data.Session.Station);
    if opt==1
        nSimulations=length(data.Session.U);
    elseif opt==2
        nSimulations=length(data.Simulation.U);
    end
    c=1;
    for paraID=1:nSimulations
        for stationID=1:nStation
            % Plot
            core_render_contour(data,...
                                dataRange,...
                                paraID,...
                                stationID,...
                                contourVar,...
                                tag,...
                                opt,...
                                nominalFlag,...
                                contourFlag);
            % Reset         
            data.database.Input=inputData0;
            caxis(data.Axes3D.Axes,[dataRange(1) dataRange(2)])

            title_text=sprintf('Sample generation[%g] @Stage[%g]',paraID, stationID);
            if ~isempty(data.logPanel)
                st=get(data.logPanel,'string');
                st{end+1}=title_text;
                set(data.logPanel, 'string',st);
            else
                disp(title_text);
            end
  
            if animationDelay<0
                pause()
            else
                pause(animationDelay)
            end
            if paraID==1
                xlim=get(data.Axes3D.Axes,'xlim');
                ylim=get(data.Axes3D.Axes,'ylim');
                zlim=get(data.Axes3D.Axes,'zlim');
            end

            if c<nSimulations*nStation
                set(data.Axes3D.Axes,'xlim',xlim)
                set(data.Axes3D.Axes,'ylim',ylim)
                set(data.Axes3D.Axes,'zlim',zlim)

                reset_rendering(data, 2);
            end

            c=c+1;
        end
    end

else
    core_render_contour(data,...
                         dataRange,...
                         paraID,...
                         stationID,...
                         contourVar,...
                         tag,...
                         opt,...
                         nominalFlag,...
                         contourFlag);
    caxis(data.Axes3D.Axes,[dataRange(1) dataRange(2)])

    title_text=sprintf('Sample generation[%g] @Stage[%g]',paraID, stationID);
    if ~isempty(data.logPanel)
        st=get(data.logPanel,'string');
        st{end+1}=title_text;
        set(data.logPanel, 'string',st);
    else
        disp(title_text);
    end
end                      
%
%--------------
function core_render_contour(data,...
                             dataRange,...
                             paraID,...
                             stationID,...
                             contourVar,...
                             tag,...
                             opt,...
                             nominalFlag,...
                             contourFlag)
% Plot inputs
if opt==1
    data.database.Input=data.Session.Input{paraID}{stationID};
elseif opt==2
    data.database.Input=data.Simulation.Input{paraID}{stationID};
end
%
for id=data.Session.Station(stationID).Part
    data.database.Input.Part(id).Graphic.Show=nominalFlag(id);   
end
modelPlotDataGeom(data, 'Part', tag);
%
for id=data.Session.Station(stationID).Part
    if data.database.Input.Part(id).Graphic.Show
        if data.database.Input.Part(id).Geometry.Type{1}==2 % morphed
            ncontrol=length(data.database.Input.Part(id).Morphing);
            for k=1:ncontrol
                morphPlotDomainSingle(data, id, k, [], tag)
            end
        end
    end
end
%
plotDataInput(data, 1, tag);
%
% Plot contour plot
%
flag123=false;
flag4=false;
if opt==1
    flag123=check_results(data.Session.U, paraID);
    if flag123
        data.database.Model.Nominal.Sol.U=sum(data.Session.U{paraID}(:,1:stationID),2);
    end
    flag4=check_results(data.Session.Gap, paraID);
    if flag4
        data.database.Model.Nominal.Sol.Gap=data.Session.Gap{paraID}{stationID};
    end
elseif opt==2
    flag123=check_results(data.Simulation.U, paraID);
    if flag123
        data.database.Model.Nominal.Sol.U=sum(data.Simulation.U{paraID}(:,1:stationID),2);
    end
    flag4=check_results(data.Simulation.Gap, paraID);
    if flag4
        data.database.Model.Nominal.Sol.Gap=data.Simulation.Gap{paraID}{stationID};
    end
end
%
% Variable to plot  
if contourVar==1
    data.database.Model.Nominal.Post.Contour.ContourVariable='u'; 
elseif contourVar==2
    data.database.Model.Nominal.Post.Contour.ContourVariable='v'; 
elseif contourVar==3
    data.database.Model.Nominal.Post.Contour.ContourVariable='w'; 
elseif contourVar==4
    data.database.Model.Nominal.Post.Contour.ContourVariable='gap'; 
end
%
% Plot contour
if contourVar<=3
    data.database.Model.Nominal.Post.Contour.MaxRange=dataRange(2); % max color range
    data.database.Model.Nominal.Post.Contour.MinRange=dataRange(1);  % min color range
    for id=data.Session.Station(stationID).Part
        if contourFlag(id)
            
            data.database.Model.Nominal.Post.Contour.Domain=id; 

            if flag123
                contourPlot(data.database.Model.Nominal, tag)
            end
        end
    end
elseif contourVar==4  % gap
    for id=data.Session.Station(stationID).Part
        c=1;
        flag_p=false;
        for idc=data.Session.Station(stationID).Contact
            idSlave=data.database.Input.Contact(idc).Slave;
            
            if id==idSlave % plot contact pair
                data.database.Model.Nominal.Post.Contour.MaxRange=dataRange(2); % max color range
                data.database.Model.Nominal.Post.Contour.MinRange=dataRange(1);  % min color range
                data.database.Model.Nominal.Post.Contour.Domain=idSlave; 
                data.database.Model.Nominal.Post.Contour.ContactPair=c; 
                c=c+1;
                if flag4
                    contourPlot(data.database.Model.Nominal, tag)
                end
                flag_p=true;
            end
        end
        
        if ~flag_p % plot only the deformed part
            data.database.Model.Nominal.Post.Contour.ContourVariable='u'; % this is only to use "U" for the deformation 
            data.database.Model.Nominal.Post.Contour.MaxRange=0.0; % max color range
            data.database.Model.Nominal.Post.Contour.MinRange=0.0;  % min color range
            data.database.Model.Nominal.Post.Contour.Domain=id; 
            if flag123
                contourPlot(data.database.Model.Nominal, tag)
            end

        end
    end
         
end

%--
function flag=check_results(var, paraID)

flag=true;
if iscell(var)
    if isempty(var{paraID})
        flag=false;
    end
else
    if isempty(var)
        flag=false;
    end
end
