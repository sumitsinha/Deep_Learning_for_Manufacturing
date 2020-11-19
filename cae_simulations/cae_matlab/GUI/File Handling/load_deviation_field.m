function load_deviation_field(source, event, h, opt)

data=guidata(h);

% pop-up menu
st = inputdlg('Enter Station ID:','Station input');
if isempty(st)
   return
end
st=str2double(st{1});
checkcond.b(1)=1;
checkcond.Type{1}='>=';
checkcond.b(2)=length(data.Session.Station);
checkcond.Type{2}='<=';
[stationID, flag]=check_condition_double(st, 1, checkcond);
if ~flag
    st=get(data.logPanel,'string');
    st{end+1}=sprintf('Error: invalid station ID!');
    set(data.logPanel, 'string',st);
    return
else
    st=get(data.logPanel,'string');
    st{end+1}=sprintf('Message: working on station ID: %g', stationID);
    set(data.logPanel, 'string',st);
end
%--
[file, path] = uigetfile({'*.csv';'*.dat';'*.'},'Open deviation field...');
%--
if file>0
    filepath=[path, file];
    maxcol=size(data.database.Model.Nominal.xMesh.Node.Coordinate,1);
    p=modelLoadInputFile(filepath, maxcol, false);
    if ~isempty(p)
        if size(p,2)==maxcol
            dfield=p(:,1:end); 
        elseif size(p,2)==(maxcol+1)
            dfield=p(:,1:end-1); % remove last column with flag status
        else
            error('Wrong file format @%s', filepath)
        end
        nSimulations=size(dfield,1);
        nSimPreAllocated=length(data.Simulation.U);
        for i=1:nSimulations
            if nSimPreAllocated<i
                data.Simulation.U{i}=data.Simulation.U{i-1};
                data.Simulation.Input{i}=data.Simulation.Input{i-1};
            end
            data.Simulation.U{i}(opt:6:end,stationID)=dfield(i,:);
        end
        %--
        st=get(data.logPanel,'string');
        st{end+1}=sprintf('Message: no. of simulations loaded: %g', nSimulations);
        set(data.logPanel, 'string',st);
        %--
        % save back
        guidata(h, data);
    end
end
