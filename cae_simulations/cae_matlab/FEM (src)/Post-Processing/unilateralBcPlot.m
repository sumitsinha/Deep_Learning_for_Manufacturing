% plot unilateral constraints
function unilateralBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.Unilateral);

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
    
     P0=fem.Boundary.Constraint.Unilateral(i).Pm;
     N0=fem.Boundary.Constraint.Unilateral(i).Nm;

     offset=fem.Boundary.Constraint.Unilateral(i).Offset;
     
     Pm=[Pm;P0];
     Nm=[Nm;N0];
     
     % create object
     [X,Y,Z]=createCylObj(radius,-N0,P0+offset*N0);
     
     if isfield(fem.Boundary.Constraint.Unilateral(i), 'Type')
     
         tpe=fem.Boundary.Constraint.Unilateral(i).Type{1};
         
          % plot symbol
          if strcmp(tpe,'assigned') %% green color
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
          
     else
         
             patch(surf2patch(X,Y,Z),...
              'facecolor','r',...
              'edgecolor','k',...
              'parent',fem.Post.Options.ParentAxes)            
     end
      
     % plot projection points
     if projptn
         
         tpe=fem.Boundary.Constraint.Unilateral(i).Type;

         for j=1:length(tpe)

             if strcmp(tpe{j},'assigned')

                 Pmi=fem.Boundary.Constraint.Unilateral(i).Pms(j,:);

                 Pmp=[Pmp;Pmi];

             end

         end
     end
     
end

% plot arrow
quiver3(Pm(:,1),Pm(:,2),Pm(:,3),Nm(:,1),Nm(:,2),Nm(:,3),L,...
        'color','b',...
        'linewidth',1,...
        'parent',fem.Post.Options.ParentAxes);
    
% plot projected
if projptn
    
    if ~isempty(Pmp)
        
        line('xdata',Pmp(:,1),...
          'ydata',Pmp(:,2),...
          'zdata',Pmp(:,3),...
          'marker','o',...
          'markersize',radius,...
          'markerfacecolor','g',...
          'linestyle','none',...
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



