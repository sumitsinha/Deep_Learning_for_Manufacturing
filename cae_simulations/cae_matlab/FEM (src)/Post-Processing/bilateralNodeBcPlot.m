% plot unilateral constraints
function bilateralNodeBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.Bilateral.Node);

%-
if nc==0 
    return
end

%-
radius=fem.Post.Options.SymbolSize;

Pma=[];
Pmna=[];
for i=1:nc
    
    nid=length(fem.Boundary.Constraint.Bilateral.Node(i).Node);
    for j=1:nid
        
        st=fem.Boundary.Constraint.Bilateral.Node(i).Type{j};
        id=fem.Boundary.Constraint.Bilateral.Node(i).Node(j);
        
        if strcmp(st,'assigned')
             P0=fem.xMesh.Node.Coordinate(id,:);
             Pma=[Pma;P0];
        else % not assigned
             P0=fem.xMesh.Node.Coordinate(id,:);
             Pmna=[Pmna;P0];
        end
         
    end
     
end

% plot Pm assigned
if ~isempty(Pma)
    line('xdata',Pma(:,1),...
      'ydata',Pma(:,2),...
      'zdata',Pma(:,3),...
      'marker','o',...
      'markersize',radius,...
      'markerfacecolor','g',...
      'linestyle','none',...
      'parent',fem.Post.Options.ParentAxes)
end

% plot Pm not assigned
if ~isempty(Pmna)
    line('xdata',Pmna(:,1),...
      'ydata',Pmna(:,2),...
      'zdata',Pmna(:,3),...
      'marker','o',...
      'markersize',radius,...
      'markerfacecolor','r',...
      'linestyle','none',...
      'parent',fem.Post.Options.ParentAxes)
end

if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end

