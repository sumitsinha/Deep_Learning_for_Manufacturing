% plotta le terne locale agli elementi 
function normalNodePlot(fem, idpart)

L=fem.Post.Options.LengthAxis;
subs=fem.Post.Options.SubSampling; % subsampling ratio

idnode=fem.Domain(idpart).Node;    
nnode=length(idnode);

% random selection
sel = randperm(nnode);

% subs percentage
sel = sel(1:floor(end*subs));
sel=idnode(sel);

% loop over nodes
P0=zeros(length(sel),3);
N0=zeros(length(sel),3);
for i=1:length(sel)

    Por=fem.xMesh.Node.Coordinate(sel(i),:);
    Nn=fem.xMesh.Node.Normal(sel(i),:);

    P0(i,:)=Por;
    N0(i,:)=Nn;

end

    
%--
quiver3(P0(:,1),P0(:,2),P0(:,3),N0(:,1),N0(:,2),N0(:,3),L,...
    'color','r',...
    'parent',fem.Post.Options.ParentAxes);

