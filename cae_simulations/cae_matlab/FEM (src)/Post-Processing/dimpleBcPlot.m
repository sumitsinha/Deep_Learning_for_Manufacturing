% plot unilateral constraints
function dimpleBcPlot(fem)

% read number of constraints
nc=length(fem.Boundary.DimplePair);

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
            
         P0=fem.Boundary.DimplePair(i).Pm;
         
         Pm=[Pm;P0];
                
         % plot projection
         if projptn
             
         type=fem.Boundary.DimplePair(i).Type;

             if ~strcmp(type,'not-assigned')

                 % master
                 P0=fem.Boundary.DimplePair(i).Pms;
                 Pmp=[Pmp;P0];
                 
                 % slave
                 P0=fem.Boundary.DimplePair(i).Psl;
                 Pmp=[Pmp;P0];

             end
             
         end 
end

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
if projptn
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

