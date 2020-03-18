% plot unilateral constraints
function projectionUnilateralBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.Unilateral);

%-
if nc==0 
    return
end

%-
radius=fem.Post.Options.SymbolSize;

Pm=[];
for i=1:nc
    
     type=fem.Boundary.Constraint.Unilateral(i).Type;
     
     for j=1:length(type)
         
         if ~strcmp(type{j},'not-assigned')

             Pmi=fem.Boundary.Constraint.Unilateral(i).Pms(j,:);
             
             Pm=[Pm;Pmi];

         end
         
     end
     
end

%--
line('xdata',Pm(:,1),...
  'ydata',Pm(:,2),...
  'zdata',Pm(:,3),...
  'marker','o',...
  'markersize',radius,...
  'parent',fem.Post.Options.ParentAxes)
     
view(3)
axis equal

if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end


