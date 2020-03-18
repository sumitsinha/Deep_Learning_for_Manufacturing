% plot unilateral constraints
function projectionBilateralElementBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.Bilateral.Element);

%-
if nc==0 
    return
end

%-
radius=fem.Post.Options.SymbolSize;

for i=1:nc
         
         type=fem.Boundary.Constraint.Bilateral.Element(i).Type;
         
         if ~strcmp(type,'not-assigned')
             
             P0=fem.Boundary.Constraint.Bilateral.Element(i).Pms;

             % create object
             [X,Y,Z]=createSphereObj(radius,P0);

             % plot
             patch(surf2patch(X,Y,Z),...
                  'facecolor','g',...
                  'edgecolor','k',...
                  'parent',fem.Post.Options.ParentAxes)
              
         end
          
end
     
view(3)
axis equal

if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end

