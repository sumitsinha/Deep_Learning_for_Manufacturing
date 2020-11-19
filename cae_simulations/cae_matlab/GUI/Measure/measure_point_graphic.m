function measure_point_graphic(~, ~, h)

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
delete(findobj(fig, 'tag',tagProbe));

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
                      'o', 'parent',ax,'tag',tagProbe,'markerfacecolor','k','markersize',10)
                  
        % write the result
        st=get(hbin,'string');
        st{end+1}='Point:';
        st{end+1}=sprintf('%f %f %f', Ps(1), Ps(2), Ps(3));
        set(hbin, 'string',st);
                  
    else
        st=get(hbin,'string');
        st{end+1}='Error - no point picked!';
        set(hbin, 'string',st);   
    end

else
    st=get(hbin,'string');
    st{end+1}='Error - no point picked!';
    set(hbin, 'string',st);   
end

%-- deactivate commands
set(fig,'WindowButtondownFcn','')


