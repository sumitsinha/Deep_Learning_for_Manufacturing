% plot pinslot constraints
function pinslotBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.PinSlot);

%-
if nc==0 
    return
end

%-
radius=fem.Post.Options.SymbolSize;

for i=1:nc
    
     P0=fem.Boundary.Constraint.PinSlot(i).Pm;
     N1=fem.Boundary.Constraint.PinSlot(i).Nm1;
     N2=fem.Boundary.Constraint.PinSlot(i).Nm2;

     tr=fem.Boundary.Constraint.PinSlot(i).Value(1);
     teta=fem.Boundary.Constraint.PinSlot(i).Value(2);
     
     % create object
     [X,Y,Z]=createSlotObj(radius,2*radius, N1, N2, P0, tr, teta);
    
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

idcon=fem.Boundary.Constraint.PinSlot(id).ConstraintId;

for i=idcon
    
    type=fem.Boundary.Constraint.Bilateral.Element(i).Type;
    
    if strcmp(type,'not-assigned')
        flag=false;
        break
    end
    
end



