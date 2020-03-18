function labelElementPlot(fem, idpart)
 
idele=fem.Domain(idpart).Element;  
nele=length(idele);

v=zeros(nele,3);
s=cell(1,nele);
for i=1:nele  

    % calcolo il baricentro dei nodi dell'elemento
    idnodes=fem.xMesh.Element(idele(i)).Element; 
    P=fem.xMesh.Node.Coordinate(idnodes,:);
    vi=mean(P,1);

    str=sprintf('%u',idele(i));
    
    v(i,:)=vi;
    
    s{i}=str;

end

text(v(:,1),...
     v(:,2),...
     v(:,3),...
     s,...
     'fontsize',fem.Post.Options.LabelSize,...
     'parent',fem.Post.Options.ParentAxes)
   
