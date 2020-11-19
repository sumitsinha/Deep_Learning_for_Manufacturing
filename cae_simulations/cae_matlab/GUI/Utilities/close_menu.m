function close_menu(source, event, h)

data=guidata(h);

n=length(data.setPanel);

for i=1:n
    delete(get(data.setPanel(i),'children'))
end
