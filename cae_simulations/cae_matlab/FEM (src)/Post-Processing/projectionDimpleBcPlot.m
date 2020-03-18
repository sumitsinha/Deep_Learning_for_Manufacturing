% plot unilateral constraints
function projectionDimpleBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.DimplePair);

%-
if nc==0 
    return
end

%-
radius=fem.Post.Options.SymbolSize;

for i=1:nc
         
         type=fem.Boundary.DimplePair(i).Type;
         
         if ~strcmp(type,'not-assigned')
             
             P0=fem.Boundary.DimplePair(i).Psl;

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

