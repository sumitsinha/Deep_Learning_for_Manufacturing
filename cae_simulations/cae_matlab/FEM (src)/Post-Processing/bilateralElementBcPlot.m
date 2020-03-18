% plot unilateral constraints
function bilateralElementBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.Bilateral.Element);

%-
if nc==0 
    return
end

%-
radius=fem.Post.Options.SymbolSize;
projptn=fem.Post.Options.ShowProjection;

Pm=[];
Pmp=[];
for i=1:nc
            
         P0=fem.Boundary.Constraint.Bilateral.Element(i).Pm;
         
         Pm=[Pm;P0];
                
         % plot projection
         if projptn
             
         type=fem.Boundary.Constraint.Bilateral.Element(i).Type;

             if ~strcmp(type,'not-assigned')

                 P0=fem.Boundary.Constraint.Bilateral.Element(i).Pms;
                 
                 Pmp=[Pmp;P0];

             end
             
         end
                
end

        % % plot Pm
        % line('xdata',Pm(:,1),...
        %   'ydata',Pm(:,2),...
        %   'zdata',Pm(:,3),...
        %   'marker','o',...
        %   'markersize',radius,...
        %   'markerfacecolor','r',...
        %   'linestyle','none',...
        %   'parent',fem.Post.Options.ParentAxes)

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

