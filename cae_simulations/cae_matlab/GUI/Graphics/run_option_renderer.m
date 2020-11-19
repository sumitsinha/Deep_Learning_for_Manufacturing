function run_option_renderer(source, event, h)

data=guidata(h);

%--
font_size=data.Options.FontSize;
%---------

% delete previous objs
delete(get(data.setPanel,'children'))

% model settings
panv=uipanel('Parent',data.setPanel,'Units', 'normalized','Position',[0.02,0.1,0.95,0.9],...
            'Title','Model settings',...
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

% show axes
uicontrol( 'Parent', bh1, 'String', ['Show Axes','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
uicontrol( 'Parent', bh1,'Callback','','style','checkbox','value',data.Axes3D.Options.ShowAxes,...
               'tag','edit1',...
               'FontSize',font_size)
  
% show frame
uicontrol( 'Parent', bh2, 'String', ['Show GCS','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
uicontrol( 'Parent', bh2,'Callback','','style','checkbox','value',data.Axes3D.Options.ShowFrame,...
               'tag','edit2',...
               'FontSize',font_size)
           
% renderer axes
uicontrol( 'Parent', bh3, 'String', ['Renderer','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
uicontrol( 'Parent', bh3,'Callback','','style','popupmenu','string',data.Axes3D.Options.Renderer(2:end),...
          'value',data.Axes3D.Options.Renderer{1},...
               'tag','edit3',...
               'FontSize',font_size,...
               'backgroundcolor','w')

% SymbolSize
uicontrol( 'Parent', bh4, 'String', ['Symbol Size','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
uicontrol( 'Parent', bh4,'Callback','','style','edit','string',data.Axes3D.Options.SymbolSize,...
               'tag','edit4',...
               'FontSize',font_size,...
               'backgroundcolor','w')
           
% LengthAxis
uicontrol( 'Parent', bh5, 'String', ['Length Axis','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
uicontrol( 'Parent', bh5,'Callback','','style','edit','string',data.Axes3D.Options.LengthAxis,...
               'tag','edit5',...
               'FontSize',font_size,...
               'backgroundcolor','w')
 
% SubSampling
uicontrol( 'Parent', bh6, 'String', ['Camera Rotation Sub-Sampling','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
uicontrol( 'Parent', bh6,'Callback','','style','edit','string',data.Axes3D.Options.SubSampling,...
               'tag','edit6',...
               'FontSize',font_size,...
               'backgroundcolor','w')

% Rotation speed
uicontrol( 'Parent', bh7, 'String', ['Camera Rotation Speed','   '],'Callback','','style','text',...
       'HorizontalAlignment','right',...
       'FontSize',font_size)   
uicontrol( 'Parent', bh7,'Callback','','style','edit','string',data.Axes3D.Options.SpeedCameraMotion,...
               'tag','edit7',...
               'FontSize',font_size,...
               'backgroundcolor','w')
           
% color figure
uicontrol( 'Parent', bh8, 'String', ['Background Color','   '],'Callback','','style','text',...
       'FontSize',font_size,...
       'HorizontalAlignment','right')

uicontrol( 'Parent', bh8, 'String', '','Callback',@set_color,'style','pushbutton',...
       'FontSize',font_size,...
       'backgroundcolor',data.Axes3D.Options.Color,...
       'HorizontalAlignment','right',...
       'tag','editcolor')

% button settings
b = uiextras.HButtonBox('Parent', data.setPanel, 'Units', 'normalized','Position',[0,0.0,1,0.1]);

uicontrol( 'Parent', b, 'String', 'Save','Callback', {@apply_option_renderer, h}, 'FontSize',font_size)
uicontrol( 'Parent', b, 'String', 'Cancel','Callback', {@close_menu, h}, 'FontSize',font_size)

set(b,'ButtonSize',[100 50])


%--
function set_color(source, event)

c=uisetcolor();

if length(c)>1
    set(source, 'backgroundcolor', c)   
end


function apply_option_renderer(source, event, h)

checkcond.Enable=true;

data=guidata(h);

% 1
obj=findobj(h,'tag','edit1');
val=get(obj,'value');

data.Axes3D.Options.ShowAxes=val;
if val
    set(data.Axes3D.Axes,'visible','on')
else
    set(data.Axes3D.Axes,'visible','off')
end

% 2
obj=findobj(h,'tag','edit2');
val=get(obj,'value');

data.Axes3D.Options.ShowFrame=val;
if val
    
    % delete previuos
    obj=findobj(data.Figure(1),'tag','tempGCS');
    delete(obj)
    
    %--
    lsymbol=data.Axes3D.Options.LengthAxis;
    plotFrame(eye(3,3), [0 0 0], data.Axes3D.AxesGCS, lsymbol,'tempGCS')
    
    set(data.Axes3D.AxesGCS,'xlim',[-lsymbol/2-0.1*lsymbol lsymbol+0.1*lsymbol]);
    set(data.Axes3D.AxesGCS,'ylim',[-lsymbol/2-0.1*lsymbol lsymbol+0.1*lsymbol]);
    set(data.Axes3D.AxesGCS,'zlim',[-lsymbol/2-0.1*lsymbol lsymbol+0.1*lsymbol]);

else
    obj=findobj(data.Figure(1),'tag','tempGCS');
    delete(obj)
end

% 3
obj=findobj(h,'tag','edit3');
val=get(obj,'value');

data.Axes3D.Options.Renderer{1}=val;
set(data.Figure(1), 'Renderer',data.Axes3D.Options.Renderer{val+1})

% 3
obj=findobj(h,'tag','edit4');
val=get(obj,'string');
    
checkcond.b=0;
checkcond.Type={'>'};
[val, flag]=check_input_double_edit(val, data.Axes3D.Options.SymbolSize,checkcond);

 if ~flag
    st=get(data.logPanel,'string');
    st{end+1}='Warning - using default value!';
    set(data.logPanel, 'string',st);
 else
    data.Axes3D.Options.SymbolSize=val; 
 end
     
    
 % 4
obj=findobj(h,'tag','edit5');
val=get(obj,'string');
    
checkcond.b=0;
checkcond.Type={'>'};
[val, flag]=check_input_double_edit(val, data.Axes3D.Options.LengthAxis,checkcond);

 if ~flag
    st=get(data.logPanel,'string');
    st{end+1}='Warning - using default value!';
    set(data.logPanel, 'string',st);
 else
    data.Axes3D.Options.LengthAxis=val; 
 end
 
 % 6
obj=findobj(h,'tag','edit6');
val=get(obj,'string');
    
checkcond.b=[0 1];
checkcond.Type={'>' '<='};
[val, flag]=check_input_double_edit(val, data.Axes3D.Options.SubSampling,checkcond);

 if ~flag
    st=get(data.logPanel,'string');
    st{end+1}='Warning - using default value!';
    set(data.logPanel, 'string',st);
 else
    data.Axes3D.Options.SubSampling=val; 
 end
 
 % 7
obj=findobj(h,'tag','edit7');
val=get(obj,'string');
    
checkcond.b=0;
checkcond.Type={'>'};
[val, flag]=check_input_double_edit(val, data.Axes3D.Options.SpeedCameraMotion,checkcond);

 if ~flag
    st=get(data.logPanel,'string');
    st{end+1}='Warning - using default value!';
    set(data.logPanel, 'string',st);
 else
    data.Axes3D.Options.SpeedCameraMotion=val; 
 end
 
 % color
obj=findobj(h,'tag','editcolor');
c=get(obj,'backgroundcolor');
data.Axes3D.Options.Color=c;

set(data.frame3d,'backgroundcolor', c)
set(data.frame3dgcs,'backgroundcolor', c)
   
 
%--
guidata(h, data);
