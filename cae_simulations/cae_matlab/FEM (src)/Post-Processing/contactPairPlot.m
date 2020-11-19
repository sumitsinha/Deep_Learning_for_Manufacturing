% plot contact point
function contactPairPlot(fem, idpair)

% marker size
radius=fem.Post.Options.SymbolSize;

Pm=fem.Boundary.ContactPair(idpair).Pms;

Ps=fem.Boundary.ContactPair(idpair).Psl;
 
% plot
if ~isempty(Pm)
    
    line('xdata',Pm(:,1),...
      'ydata',Pm(:,2),...
      'zdata',Pm(:,3),...
      'marker','x',...
      'markersize',radius,...
      'markerfacecolor','g',...
      'linestyle','none',...
      'parent',fem.Post.Options.ParentAxes)
  
end

% plot
if ~isempty(Ps)
    
    line('xdata',Ps(:,1),...
      'ydata',Ps(:,2),...
      'zdata',Ps(:,3),...
      'marker','o',...
      'markersize',radius,...
      'markerfacecolor','b',...
      'linestyle','none',...
      'parent',fem.Post.Options.ParentAxes)
  
end


view(3)
axis equal

if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end


