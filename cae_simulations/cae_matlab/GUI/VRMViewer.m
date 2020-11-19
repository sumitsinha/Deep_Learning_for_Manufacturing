% Main GUI Viewer
function VRMViewer

%------------------
clc
% clear
close all
warning('OFF', 'all');
%------------------

        %% Import libraries
        % import javax.swing.*
        % import javax.swing.tree.*;
        %------------------------------------ 
        %------------------------------------ 

%% Define properties
release='@2.0.0';
name_tool=['VRM - Viewer',release];
back_color=[0.8 0.8 0.8];
font_size=10;
%
%% Define main figure
fig=figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'menubar','none',...
    'Name',name_tool,'NumberTitle','off','dockcontrol','off','resize','on',...
     'CloseRequestFcn',@close_main_fig);
% Vertical split
hhflex = uix.HBoxFlex( 'Parent', fig ,'position',[0 0 1 1]);
% Panel main[1]
bx1=uiextras.BoxPanel( 'Parent', hhflex, 'Units', 'normalized', 'Title', 'Settings',...
    'BorderType','etchedout','ShadowColor','k','ForegroundColor','k','Tag','setModel',...
    'FontSize',font_size,'FontWeight','normal');
% Panel main[2]
bx2=uiextras.BoxPanel( 'Parent', hhflex, 'Units', 'normalized', 'Title', '3D Renderer',...
    'BorderType','etchedout','ShadowColor','k','ForegroundColor','k','Tag','setRenderer',...
    'FontSize',font_size,'FontWeight','normal');
set( hhflex, 'Widths', [-1 -3], 'Spacing', 5 );
% Horizontal split
hvflex = uix.VBoxFlex( 'Parent', bx1);
setPanel=uipanel('Parent',hvflex,'Units', 'normalized','Position',[0 0 1 1]);
% Log panel
logPanel=uiextras.BoxPanel( 'Parent', hvflex, 'Units', 'normalized', 'Title', 'Log Panel',...
    'BorderType','etchedout','ShadowColor','k','ForegroundColor','k','Tag','logPanel',...
    'FontSize',font_size,'FontWeight','normal');
logcontr=uicontrol('Parent', logPanel, 'Units', 'normalized', 'Position', [0 0 1 1],...
    'Tag','logControl','FontSize',font_size,'FontWeight','normal',...
    'style','listbox','BackgroundColor','w',...
    'String',{'---'});
set( hvflex, 'Heights', [-3 -1], 'Spacing', 5 );
% define context menus
hm = uicontextmenu;
uimenu(hm, 'Label', 'Clear...','Callback',{@clear_log_panel, fig}); 
set(logcontr,'UIContextMenu',hm)
%
%% Define axes
frame3d=uipanel('Parent',bx2,'Units','normalized','Position',[0 0.2 1 0.8],'backgroundcolor',back_color);
frame3dgcs=uipanel('Parent',frame3d,'Units','normalized','Position',[0 0 0.2 0.2],'backgroundcolor',back_color,...
           'BorderType','none');
haxes3d=axes('Parent',frame3d,'Units','normalized','Position',[0 0 1 1],'Box','on','Visible','off', 'clipping', 'off');
axis equal, hold all, view(3), grid on
camlight('headlight')
material dull
% GCS
haxes3dgcs=axes('Parent',frame3dgcs,'Units','normalized','Position',[0 0 1 1],'Box','on','Visible','off');
axis equal, hold all, view(3), grid on 
contourcmap('jet')
colorbar('peer',haxes3d,...
         'units','normalized',...
         'position',[0.05, 0.5, 0.05, 0.4],...
         'UIContextMenu','') 
%
%% Define menus
m=uimenu(fig,'Label','File');
     uimenu(m, 'Label', 'Select Session Folder','Callback',{@open_database, fig});
     uimenu(m, 'Label', 'Load Session','Callback',{@run_load_model, fig});  
     hm=uimenu(m,'Label','Export','Callback','','Separator','on');
        uimenu(hm,'Label','Session','Callback',{@export_dataset, fig, 1}); 
        uimenu(hm,'Label','Simulation','Callback',{@export_dataset, fig, 2}); 
     uimenu(m,'Label','Export Screenshoot','Callback',{@export_image, fig}); 
m=uimenu(fig,'Label','Navigate');
   uimenu(m,'Label','Rotate','Callback',{@run_navigate, fig, 'rotate'});  
   uimenu(m,'Label','Zoom','Callback',{@run_navigate, fig, 'zoom'});  
   uimenu(m,'Label','Pan','Callback',{@run_navigate, fig, 'pan'}); 
   vw=uimenu(m,'Label','View');  
   uimenu(vw,'Label','XY','Callback',{@run_setview, fig, 'XY'});  
   uimenu(vw,'Label','XZ','Callback',{@run_setview, fig, 'XZ'});  
   uimenu(vw,'Label','YZ','Callback',{@run_setview, fig, 'YZ'});   
   uimenu(vw,'Label','YX','Callback',{@run_setview, fig, 'YX'});  
   uimenu(vw,'Label','ZX','Callback',{@run_setview, fig, 'ZX'});  
   uimenu(vw,'Label','ZY','Callback',{@run_setview, fig, 'ZY'});  
   uimenu(vw,'Label','Iso view','Callback',{@run_setview, fig, 'Iso'});    
   uimenu(m,'Label','Select','Callback',{@run_navigate, fig, 'select'},'Separator','on'); 
m=uimenu(fig,'Label','Tools');
   hm=uimenu(m, 'Label', 'Measure');  
       uimenu(hm, 'Label', 'Point', 'Callback',{@measure_point_graphic, fig});  
       uimenu(hm, 'Label', 'Distance', 'Callback',{@measure_distance_graphic, fig});  
m=uimenu(fig,'Label','Graphics');
   uimenu(m, 'Label', 'Options','Callback',{@run_option_renderer, fig});  
m=uimenu(fig,'Label','Simulation');
   uimenu(m, 'Label', 'Import process parameters...', 'Callback',...
           {@load_file_table, fig})
    cm=uimenu(m, 'Label', 'Import simulation data...');
        uimenu(cm, 'Label', 'X field', 'Callback',{@load_deviation_field, fig, 1});
        uimenu(cm, 'Label', 'Y field', 'Callback',{@load_deviation_field, fig, 2});
        uimenu(cm, 'Label', 'Z field', 'Callback',{@load_deviation_field, fig, 3});
    uimenu(m, 'Label', 'Solve...','Callback',{@run_simulation_gui, fig});
        
m=uimenu(fig,'Label','Viewer');
   hm=uimenu(m, 'Label', 'Clear 3D Renderer'); 
        uimenu(hm, 'Label', 'All','Callback',{@run_clear_view, fig, 1});
        uimenu(hm, 'Label', 'Temporary Objects','Callback',{@run_clear_view, fig, 2});
        uimenu(hm, 'Label', 'Temporary Probes','Callback',{@run_clear_view, fig, 4});
        uimenu(hm, 'Label', 'Model Objects','Callback',{@run_clear_view, fig, 3});
   hm=uimenu(m, 'Label', 'Render Model');  
        uimenu(hm, 'Label', 'Show','Callback',{@run_render_model, fig,1});  
        uimenu(hm, 'Label', 'Hide','Callback',{@run_render_model, fig,2});  
   hm=uimenu(m, 'Label', 'Render');  
        uimenu(hm, 'Label', 'Session','Callback',{@run_show_results, fig, 1});  
        uimenu(hm, 'Label', 'Simulation','Callback',{@run_show_results, fig, 2});  
%
%% Save all
data=initModel();
%
% Option for rendering
data.Figure=fig;
data.Axes3D.Axes=haxes3d;
data.Axes3D.AxesGCS=haxes3dgcs;
%
data.Axes3D.Options.ShowAxes=false;
data.Axes3D.Options.Color=back_color;
data.Axes3D.Options.ShowFrame=true;
%
% Handles
data.setPanel=setPanel;
data.logPanel=logcontr;
data.frame3d=frame3d;
data.frame3dgcs=frame3dgcs;
%
% Generic options
data.Options.FontSize=font_size;
%
% plot GCS
initGCS(data);
%
% save into GUI
guidata(fig, data);
% 