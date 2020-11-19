function cell_selected_table(source, event, h)

data=guidata(h);

data.Table.Selection=event.Indices;

guidata(h,data);