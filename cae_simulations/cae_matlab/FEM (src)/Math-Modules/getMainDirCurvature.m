% calculate principal direction of curvature
function dircurv=getMainDirCurvature(fem)

% no. of nodes
nnode=size(fem.xMesh.Node.Coordinate,1);

% set initial values
dircurv=zeros(nnode,2);
for i=1:nnode
   pi=fem.xMesh.Node.Coordinate(i,:);
   ni=fem.xMesh.Node.Normal(i,:);
   
   % get rotation matrix
   Ri=vector2Rotation(ni);

   % read elements ids connected to node i-th
   idelenear=fem.Sol.Node2Element{i};
   
   % get related nodes ids
   n=length(idelenear);
   
   idnode=[];
   for j=1:n
        idnode=[idnode fem.xMesh.Element(idelenear(j)).Element];
   end
   
   idnode=unique(idnode);
   
   % ... then coordinates
   pnear=fem.xMesh.Node.Coordinate(idnode,:);
   
   % tarsform all point into local frame
   pnear=applyinv4x4(pnear, Ri, pi);
   x=pnear(:,1); y=pnear(:,2); z=pnear(:,3);

   % calculate best fitting of parabolid: a*x^2 + b*y^2 = z
   A=[x.^2 2*x.*y y.^2];
   b=z;
   
   % pseudoinverse
   C=A\b;
   
   % ... finally gaussian curvature  
   b=-(C(1) + C(3)); c=C(1)*C(3)-C(2)^2;
   
   k1=(-b+sqrt(b^2-4*c))/2;
   k2=(b+sqrt(b^2-4*c))/2;
   
   % save back
   dircurv(i,:)=[k1 k2];
   
end

