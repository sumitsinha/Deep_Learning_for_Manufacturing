% plot 
function log_current_selection(src, ~, logPanel)

%--
if isempty(logPanel)
    return
end

% get obj tag
tag=get(src, 'tag');

st=get(logPanel,'string');
st{end+1}=sprintf('Current selection: %s',tag);

set(logPanel, 'string',st);  

