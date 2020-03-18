% plot normal to elements
function normalElementPlot(fem, idpart, tag)

if nargin==2
    tag='';
end

L=fem.Post.Options.LengthAxis;
subs=fem.Post.Options.SubSampling; % subsampling ratio
    
idele=fem.Domain(idpart).Element;  

nele=length(idele);

% random selection
sel = randperm(nele);

% subs percentage
sel = sel(1:floor(end*subs));
sel=idele(sel);

% loop over nodes
P0=zeros(length(sel),3);
N0=zeros(length(sel),3);
for i=1:length(sel)

    idnodeele=fem.xMesh.Element(sel(i)).Element; 

    Pi=fem.xMesh.Node.Coordinate(idnodeele,:);
    Por=mean(Pi,1);

    Nn=fem.xMesh.Element(sel(i)).Tmatrix.Normal;

    P0(i,:)=Por;
    N0(i,:)=Nn;
    
end
    
renderAxis(N0, P0, fem.Post.Options.ParentAxes, L, tag,'k');
    




