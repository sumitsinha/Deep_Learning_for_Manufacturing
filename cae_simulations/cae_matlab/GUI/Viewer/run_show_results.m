function run_show_results(source, event, h, opt)

% opt 1: => render session (data.Session)
% opt 2: => render simulation (data.Simulation)

data=guidata(h);

if opt==1
    flagd=data.Session.Flag;
elseif opt==2
    flagd=data.Simulation.Flag;
end
%
if ~flagd
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
% model settings
panv=uipanel('Parent',data.setPanel,'Units', 'normalized','Position',[0.02,0.35,0.95,0.65],...
            'Title','Model settings',...
            'FontSize',font_size);
  
pantable=uipanel('Parent',data.setPanel,'Units', 'normalized','Position',[0.02,0.1,0.95,0.25],...
            'Title','Model rendering',...
            'FontSize',font_size);
        
bv = uiextras.VButtonBox('Parent', panv, 'Units', 'normalized','Position',[0.01,0,0.95,1],...
 'HorizontalAlignment','right','VerticalAlignment','top');
set(bv,'ButtonSize',[400 30])

bh1 = uiextras.HButtonBox('Parent', bv,...
    'HorizontalAlignment','right','VerticalAlignment','top');
set(bh1,'ButtonSize',[400 30])

bh2 = uiextras.HButtonBox('Parent', bv,...
    'HorizontalAlignment','right','VerticalAlignment','top');
set(bh2,'ButtonSize',[400 30])
    
bh3 = uiextras.HButtonBox('Parent', bv,...
    'HorizontalAlignment','right','VerticalAlignment','top');
set(bh3,'ButtonSize',[400 30])

bh4 = uiextras.HButtonBox('Parent', bv,...
    'HorizontalAlignment','right','VerticalAlignment','top');
set(bh4,'ButtonSize',[400 30])

bh5 = uiextras.HButtonBox('Parent', bv,...
    'HorizontalAlignment','right','VerticalAlignment','top');
set(bh5,'ButtonSize',[400 30])

bh6 = uiextras.HButtonBox('Parent', bv,...
    'HorizontalAlignment','right','VerticalAlignment','top');
set(bh6,'ButtonSize',[400 30])

bh7 = uiextras.HButtonBox('Parent', bv,...
    'HorizontalAlignment','right','VerticalAlignment','top');
set(bh7,'ButtonSize',[400 30])

bh8 = uiextras.HButtonBox('Parent', bv,...
    'HorizontalAlignment','right','VerticalAlignment','top');
set(bh8,'ButtonSize',[400 30])

% parameter ID
uicontrol( 'Parent', bh1, 'String', ['Sample ID','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)
   
if opt==1
    nSimulations=length(data.Session.U);
elseif opt==2
    nSimulations=length(data.Simulation.U);
end
if nSimulations==0
    s=cell(1);
else
    s=cell(1,nSimulations);
    for i=1:nSimulations
        s{i}=sprintf('%g',i);
    end
end  
uicontrol( 'Parent', bh1,'Callback','','style','popupmenu','string',s,...
          'value',1,...
           'tag','edit1',...
           'FontSize',font_size,...
           'backgroundcolor','w')
                   
% Stages
uicontrol( 'Parent', bh2, 'String', ['Stage','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
nstation=length(data.Session.Station);
if nstation==0
    s=cell(1);
else
    s=cell(1,nstation);
    for i=1:nstation
        s{i}=sprintf('%g - %s', i, data.Session.Station(i).Type{data.Session.Station(i).Type{1}+2});
    end
end  
uicontrol( 'Parent', bh2,'Callback','','style','popupmenu','string',s,...
          'value',1,...
           'tag','edit2',...
           'FontSize',font_size,...
           'backgroundcolor','w')

% Contour variable
uicontrol( 'Parent', bh3, 'String', ['Variable','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)

uicontrol('Parent', bh3,'Callback','','style','popupmenu','string',{'Displacement X','Displacement Y','Displacement Z', 'Gap'},...
          'value',1,...
           'tag','edit3',...
           'FontSize',font_size,...
           'backgroundcolor','w')
       
% Contour limits
uicontrol( 'Parent', bh4, 'String', ['Data Range (m/M)','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)

uicontrol( 'Parent', bh4,'Callback','','style','edit','string','-2.0',.....
           'tag','edit4.1',...
           'backgroundcolor','w',...
           'FontSize',font_size) 

uicontrol( 'Parent', bh4,'Callback','','style','edit','string','2.0',.....
           'tag','edit4.2',...
           'backgroundcolor','w',...
           'FontSize',font_size) 
     
% Deformed
uicontrol( 'Parent', bh5, 'String', ['Scale/Deformed','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)  
uicontrol( 'Parent', bh5,'Callback','','style','edit','string','1.0',.....
           'tag','edit5.1',...
           'backgroundcolor','w',...
           'FontSize',font_size) 
uicontrol( 'Parent', bh5,'Callback','','style','checkbox','value',1,...
               'tag','edit5.2',...
               'FontSize',font_size)
           
% Animate
uicontrol( 'Parent', bh6, 'String', ['Delay/Animate','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
uicontrol( 'Parent', bh6,'Callback','','style','edit','string','1.0',.....
           'tag','edit6.1',...
           'backgroundcolor','w',...
           'FontSize',font_size) 
uicontrol( 'Parent', bh6,'Callback','','style','checkbox','value',0,...
               'tag','edit6.2',...
               'FontSize',font_size)
          
% Part to be probed
uicontrol( 'Parent', bh8, 'String', ['Probing part','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
nparts=data.database.Model.Nominal.Sol.nDom;
s=cell(1,nparts);
for i=1:nparts
    s{i}=sprintf('Part[%g]', i);
end 
uicontrol( 'Parent', bh8,'Callback','','style','popupmenu','string',s,...
          'value',1,...
           'tag','edit7',...
           'FontSize',font_size,...
           'backgroundcolor','w')
 
% table with part settings
cnames={'Part ID','Show Nominal', 'Show Contour'};
formcolum={'Char','Logical', 'Logical'};
s=cell(nparts, 3);
for i=1:nparts
    s{i,1}=sprintf('Part[%g]', i);
    s{i,2}=true;
    s{i,3}=true;
end 
uitable('Data',s, 'ColumnName',cnames,'parent',pantable,...
        'ColumnEditable', [false true true],'SelectionHighlight','on',...
        'tag','table', 'FontSize',font_size,...
        'Units', 'normalized','Position',[0,0,1,1],...
        'ColumnFormat', formcolum);
    
% button settings
b = uiextras.HButtonBox('Parent', data.setPanel, 'Units', 'normalized','Position',[0,0.0,1,0.1]);

uicontrol( 'Parent', b, 'String', 'Play','Callback', {@play_menu, h, opt}, 'FontSize',font_size)
uicontrol( 'Parent', b, 'String', 'Probe','Callback', {@probe_menu, h, opt}, 'FontSize',font_size)
uicontrol( 'Parent', b, 'String', 'Cancel','Callback', {@close_menu, h}, 'FontSize',font_size)
set(b,'ButtonSize',[100 50])

%
function probe_menu(event, source, h, opt)

%---------------
searchDist=10.0;
%---------------

% read data
[data,...
      parameterID,...
      stationID,...
      contourVar,...
      ~,...
      ~,...
      ~,...
      ~,...
      partID,...
      ~,...
      ~]=read_data(h);
  
%--
tag=data.Axes3D.Options.Tag.TempProbe;

% Plot contour plot
if opt==1
    U=sum(data.Session.U{parameterID}(:,1:stationID),2);
elseif opt==2
    U=sum(data.Simulation.U{parameterID}(:,1:stationID),2);
end
%
% Variable to plot  
if contourVar==1
    intepVar='u'; 
elseif contourVar==2
    intepVar='v'; 
elseif contourVar==3
    intepVar='w'; 
elseif contourVar==4
    st=get(data.logPanel,'string');
    st{end+1}='Error - "Gap" variable not supported in this version!';
    set(data.logPanel, 'string',st);
    return
end
%
measure_point_contour(data, partID, intepVar, searchDist, U, tag);    
        
%
function play_menu(event, source, h, opt)

% read data
[data,...
      parameterID,...
      stationID,...
      contourVar,...
      dataRange,...
      deformedScale,...
      deformedFlag,...
      animationDelay,...
      animateFlag,...
      ~,...
      nominalFlag,...
      contourFlag]=read_data(h);
  
% play model
render_contour(data,...
                parameterID,...
                stationID,...
                contourVar,...
                dataRange,...
                deformedScale,...
                deformedFlag,...
                animationDelay,...
                animateFlag,...
                opt,...
                nominalFlag,...
                contourFlag);
            
%--   
function [data,...
          parameterID,stationID,...
          contourVar,dataRange,...
          deformedScale,deformedFlag,...
          animationDelay,...
          animateFlag,...
          partID,...
          nominalFlag,...
          ContourFlag]=read_data(h)

checkcond.Enable=true;

data=guidata(h);

% Parameter ID
obj=findobj(h,'tag','edit1');
s=get(obj,'string');
if isempty(s{1})
    st=get(data.logPanel,'string');
    st{end+1}='Error - no dataset identified!';
    set(data.logPanel, 'string',st);
    return
end
parameterID=get(obj,'value');

% Station ID
obj=findobj(h,'tag','edit2');
s=get(obj,'string');
if isempty(s{1})
    st=get(data.logPanel,'string');
    st{end+1}='Error - no dataset identified!';
    set(data.logPanel, 'string',st);
    return
end
stationID=get(obj,'value');

% Contour variable
obj=findobj(h,'tag','edit3');
contourVar=get(obj,'value');

% Contour limits
obj=findobj(h,'tag','edit4.1');
val=get(obj,'string');
valDef=-2.0;
checkcond.b=0;
checkcond.Type={'<='};
[dataRange(1), flag]=check_input_double_edit(val, valDef,checkcond);
if ~flag
    st=get(data.logPanel,'string');
    st{end+1}='Warning - using default value for min range!';
    set(data.logPanel, 'string',st);
end
%
obj=findobj(h,'tag','edit4.2');
val=get(obj,'string');
valDef=2.0;
checkcond.b=0;
checkcond.Type={'>='};
[dataRange(2), flag]=check_input_double_edit(val, valDef,checkcond);
if ~flag
    st=get(data.logPanel,'string');
    st{end+1}='Warning - using default value for max range!';
    set(data.logPanel, 'string',st);
end
 
% Deformed
obj=findobj(h,'tag','edit5.1');
val=get(obj,'string');
valDef=2.0;
checkcond.b=0;
checkcond.Type={'>='};
[deformedScale, flag]=check_input_double_edit(val, valDef,checkcond);
if ~flag
    st=get(data.logPanel,'string');
    st{end+1}='Warning - using default value for deformation scale!';
    set(data.logPanel, 'string',st);
end
%
obj=findobj(h,'tag','edit5.2');
deformedFlag=get(obj,'value');

% Animate
obj=findobj(h,'tag','edit6.1');
val=get(obj,'string');
valDef=1.0;
checkcond.b=0;
checkcond.Type={'>='};
[animationDelay, flag]=check_input_double_edit(val, valDef,checkcond);
if ~flag
    st=get(data.logPanel,'string');
    st{end+1}='Warning - using default value for animation delay!';
    set(data.logPanel, 'string',st);
end
%
obj=findobj(h,'tag','edit6.2');
animateFlag=get(obj,'value');

% Parameter ID
obj=findobj(h,'tag','edit7');
partID=get(obj,'value');

% Nominal and contour flags
obj=findobj(h,'tag','table');
d=get(obj,'data');
nominalFlag=cell2mat(d(:,2));
ContourFlag=cell2mat(d(:,3));

