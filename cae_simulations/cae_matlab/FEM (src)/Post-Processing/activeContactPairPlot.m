% plot active contact pairs
function activeContactPairPlot(fem)

% marker size
radius=fem.Post.Options.SymbolSize;


% read number of constraints
nc=length(fem.Boundary.Constraint.ContactWset);

%-
if nc==0 
    return
end

Pm=[];
Ps=[];
for i=1:nc
     Pi=fem.Boundary.Constraint.ContactWset(i).Psl;

     Ps=[Ps;Pi];

     Pi=fem.Boundary.Constraint.ContactWset(i).Pms;

     Pm=[Pm;Pi];
end
     
% plot
if ~isempty(Pm)
    
    line('xdata',Pm(:,1),...
      'ydata',Pm(:,2),...
      'zdata',Pm(:,3),...
      'marker','o',...
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
  
if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end


