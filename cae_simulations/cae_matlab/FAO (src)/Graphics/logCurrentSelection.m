% plot 
function logCurrentSelection(src, ~, logPanel)

% get obj tag
tag=get(src, 'tag');

if ~isempty(logPanel)
    st=get(logPanel,'string');
    st{end+1}=sprintf('Current selection: %s',tag);
    set(logPanel, 'string',st);  
else
    fprintf('Current selection: %s\n',tag);
end



