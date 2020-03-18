function renderAxis(N, P, ax, lsymbol, tag, col)

n=size(P,1);
for i=1:n
    p1=P(i,:)+lsymbol*N(i,:);

    %--
    p=[P(i,:)
       p1];
  
    % plot line
    plot3(p(:,1),p(:,2),p(:,3),'-','linewidth',1,'color',col,...
                              'parent',ax,...
                              'tag',tag)
     
    % plot arrow
    fmax=lsymbol/4;
    [X, Y, Z]=renderConeObj(0, fmax, 20, p1, -N(i,:),6);
    
    surf(X, Y, Z,'facecolor',col,'edgecolor','none',...
         'parent',ax,...
         'tag',tag)
    
end

