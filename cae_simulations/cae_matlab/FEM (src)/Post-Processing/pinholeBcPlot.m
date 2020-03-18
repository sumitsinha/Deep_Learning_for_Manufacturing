% plot pinhole constraints
function pinholeBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.PinHole);

%-
if nc==0 
    return
end

%-
radius=fem.Post.Options.SymbolSize;

for i=1:nc
    
     P0=fem.Boundary.Constraint.PinHole(i).Pm;
     N1=fem.Boundary.Constraint.PinHole(i).Nm1;
     N2=fem.Boundary.Constraint.PinHole(i).Nm2;
     
     tx=fem.Boundary.Constraint.PinHole(i).Value(1);
     ty=fem.Boundary.Constraint.PinHole(i).Value(2);
     
     alfa=fem.Boundary.Constraint.PinHole(i).Value(3);
     beta=fem.Boundary.Constraint.PinHole(i).Value(4);

     % create object
     [X,Y,Z]=createPinHoleObj(radius,N1, N2, P0,tx, ty, alfa, beta);
    
      % plot symbol
      if checkAssigned(fem, i) %% green color
        patch(surf2patch(X,Y,Z),...
          'facecolor','g',...
          'edgecolor','k',...
          'parent',fem.Post.Options.ParentAxes)
      else %% red color
         patch(surf2patch(X,Y,Z),...
          'facecolor','r',...
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


function flag=checkAssigned(fem, id)

flag=true;

idcon=fem.Boundary.Constraint.PinHole(id).ConstraintId;

for i=idcon
    
    type=fem.Boundary.Constraint.Bilateral.Element(i).Type;
    
    if strcmp(type,'not-assigned')
        flag=false;
        break
    end
    
end



