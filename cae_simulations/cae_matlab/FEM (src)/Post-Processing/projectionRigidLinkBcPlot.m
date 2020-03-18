% plot unilateral constraints
function projectionRigidLinkBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.RigidLink);

%-
if nc==0 
    return
end

%-
radius=fem.Post.Options.SymbolSize;

for i=1:nc
    
    type=fem.Boundary.Constraint.RigidLink(i).Type;
         
    if ~strcmp(type,'not-assigned')
             
         Pm=fem.Boundary.Constraint.RigidLink(i).Pms;
         Ps=fem.Boundary.Constraint.RigidLink(i).Psl;
          
         line('xdata',[Pm(1),Ps(1)],...
              'ydata',[Pm(2),Ps(2)],...
              'zdata',[Pm(3),Ps(3)],...
              'marker','o',...
              'markersize',radius)
      
    end
     
end
     
view(3)
axis equal

if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end

