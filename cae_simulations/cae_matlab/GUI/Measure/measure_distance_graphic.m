function measure_distance_graphic(~, ~, h)

data=guidata(h);

% reset any control
h=rotate3d(data.Figure(1));
set(h,'enable','off');
h=pan(data.Figure(1));
set(h,'enable','off');
h=zoom(data.Figure(1));
set(h,'enable','off');

set(data.Figure(1),'WindowButtondownFcn','');
set(data.Figure(1),'WindowButtonupFcn','');
set(data.Figure(1),'WindowButtonmotionFcn','');

% inizialize selection phase
set(data.Figure(1),'WindowButtondownFcn',{@Sclick, data.Figure(1), data.Axes3D.Axes,...
                                          data.Axes3D.Options.SearchDistance,...
                                          data.logPanel, data.Axes3D.Options.Tag.TempProbe});

%-----------
function Sclick(~, ~,fig, ax, searchDist, hbin, tagProbe)

%--
delete(findobj(fig, 'tag',tagProbe))

%--
delete(findobj(fig, 'tag','tempobj-ps'))

% get all points in the current view
obj=get(ax,'children');

vertex=[];
for i=1:length(obj)
    
    tag=get(obj(i), 'type');
    
    if strcmp(get(obj(i),'visible'),'on')
        
        if strcmp(tag,'patch') % patch object
            vertex=[vertex
                   get(obj(i),'vertices')];
        elseif strcmp(tag,'line') % line object
            
            x=get(obj(i),'xdata');
            if size(x,1)==1
                x=x';
            end
            
            y=get(obj(i),'ydata');
            if size(y,1)==1
                y=y';
            end
            
            z=get(obj(i),'zdata');
            if size(z,1)==1
                z=z';
            end
            
            vertex=[vertex
                    x y z];
        end
        
    end

end

if ~isempty(vertex)

    % get closest point
    [mdist, idSelected]=getClosestPointSelection(ax, vertex);
    
    if mdist<=searchDist
        Ps=vertex(idSelected,:);
        
        plot3(Ps(1),Ps(2),Ps(3),...
                      'o', 'parent',ax,'tag','tempobj-ps','markerfacecolor','k','markersize',10)
                  
        % call mouse move function          
        set(fig,'WindowButtonMotionFcn',{@Mmove, fig, ax, vertex, searchDist, Ps, hbin, tagProbe});
                  
    end

end

%--
function Mmove(~, ~,fig, ax, vertex, searchDist, Ps, hbin, tagProbe)

delete(findobj(fig, 'tag','tempobj-pe'))

% get closest point
[mdist,idSelected]=getClosestPointSelection(ax, vertex);

Pe=[];
flag=false;
if mdist<=searchDist
    flag=true;
    Pe=vertex(idSelected,:);
    
        line('xdata',[Ps(1) Pe(1)],...
         'ydata',[Ps(2) Pe(2)],...
         'zdata',[Ps(3) Pe(3)],...
         'linestyle','-',...
         'marker','o',...
         'markerfacecolor','k','markersize',10,...
         'parent',ax,...
         'tag','tempobj-pe')
end

%-call mouse buttonUp function
set(fig,'WindowButtonUpFcn',{@endClick, fig, ax, Ps, Pe, flag, hbin, tagProbe}); 

%--
function endClick(~, ~,fig, ax, Ps, Pe, flag, hbin, tagProbe)

delete(findobj(fig, 'tag','tempobj-ps'))
delete(findobj(fig, 'tag','tempobj-pe'))

if flag
              
    % line
    line('xdata',[Ps(1) Pe(1)],...
         'ydata',[Ps(2) Pe(2)],...
         'zdata',[Ps(3) Pe(3)],...
         'linestyle','-',...
         'marker','s',...
         'markerfacecolor','k','markersize',10,...
         'parent',ax,...
         'tag',tagProbe)
     
     % write the results
     st=get(hbin,'string');
     
     st{end+1}='Point (P1):';
     st{end+1}=sprintf('%f %f %f', Ps(1), Ps(2), Ps(3));
     
     st{end+1}='Point (P2):';
     st{end+1}=sprintf('%f %f %f', Pe(1), Pe(2), Pe(3));
          
     N=(Pe-Ps)/norm(Pe-Ps);
     st{end+1}='Direction (P2 - P1):';
     st{end+1}=sprintf('%f %f %f', N(1), N(2), N(3));
     
     st{end+1}='Distance:';
     st{end+1}=sprintf('%f', norm(Pe-Ps));
     
     %--
     set(hbin, 'string',st);
     
end

% disable all controls
set(fig,'WindowButtondownFcn','');
set(fig,'WindowButtonMotionFcn','');
set(fig,'WindowButtonUpFcn','');


