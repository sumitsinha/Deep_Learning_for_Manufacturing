function reset_rendering(data, opt)

% opt:
    % 1: delete all (default)
    % 2: delete only temporary ("tempobj")
    % 3: delete only "model"
    % 4: delete only temporary ("tempprobe")

if nargin==1
    opt=1;
end
%
if opt==1
    obj=findobj(data.Figure,'tag',data.Axes3D.Options.Tag.Model);
    delete(obj);
    
    obj=findobj(data.Figure,'tag',data.Axes3D.Options.Tag.TempObject);
    delete(obj);
    
    obj=findobj(data.Figure,'tag',data.Axes3D.Options.Tag.TempProbe);
    delete(obj);
elseif opt==2
    obj=findobj(data.Figure,'tag',data.Axes3D.Options.Tag.TempObject);
    delete(obj);
elseif opt==3
    obj=findobj(data.Figure,'tag',data.Axes3D.Options.Tag.Model);
    delete(obj);
elseif opt==4
    obj=findobj(data.Figure,'tag',data.Axes3D.Options.Tag.TempProbe);
    delete(obj);
end
