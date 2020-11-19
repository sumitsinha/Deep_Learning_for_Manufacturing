function clear_log_panel(source, event, h)

data=guidata(h);

set(data.logPanel,'string',{''},'value',1)