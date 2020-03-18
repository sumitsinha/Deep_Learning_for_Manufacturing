% plot unilateral constraints
function rigidlinkBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.RigidLink);

%-
if nc==0 
    return
end

%-
radius=fem.Post.Options.SymbolSize;
L=fem.Post.Options.LengthAxis;

projptn=fem.Post.Options.ShowProjection;

Pm=[];
Nm=[];
Pmp=[];
for i=1:nc
    
     P0=fem.Boundary.Constraint.RigidLink(i).Pm;
     N0=fem.Boundary.Constraint.RigidLink(i).Nm;
     
     Pm=[Pm;P0];
     Nm=[Nm;N0];
     
     if projptn
         
            type=fem.Boundary.Constraint.RigidLink(i).Type;

             if ~strcmp(type,'not-assigned')

                 % master
                 P0=fem.Boundary.Constraint.RigidLink(i).Pms;
                 Pmp=[Pmp;P0];
                 
                 % slave
                 P0=fem.Boundary.Constraint.RigidLink(i).Psl;
                 Pmp=[Pmp;P0];

             end
             
     end
 
end

% plot arrow
quiver3(Pm(:,1),Pm(:,2),Pm(:,3),Nm(:,1),Nm(:,2),Nm(:,3),L,...
        'color','b',...
        'linewidth',1,...
        'parent',fem.Post.Options.ParentAxes);
    
% plot Pm
line('xdata',Pm(:,1),...
  'ydata',Pm(:,2),...
  'zdata',Pm(:,3),...
  'marker','o',...
  'markersize',radius,...
  'markerfacecolor','c',...
  'linestyle','none',...
  'parent',fem.Post.Options.ParentAxes)

% plot Pmp
if projptn && ~isempty(Pmp)
    line('xdata',Pmp(:,1),...
      'ydata',Pmp(:,2),...
      'zdata',Pmp(:,3),...
      'marker','o',...
      'markersize',radius,...
      'markerfacecolor','g',...
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

