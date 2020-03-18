function labelNodePlot(fem, idpart)

idnode=fem.Domain(idpart).Node;  
 
nnode=length(idnode);

v=zeros(nnode,3);
s=cell(1,nnode);
for i=1:nnode
    % coordinate del nodo i-esimo
    vi=fem.xMesh.Node.Coordinate(idnode(i),:);

    str=sprintf('%u',idnode(i));
    
    v(i,:)=vi;
    
    s{i}=str;
end

text(v(:,1),...
     v(:,2),...
     v(:,3),...
     s,...
     'fontsize',fem.Post.Options.LabelSize,...
     'parent',fem.Post.Options.ParentAxes)
