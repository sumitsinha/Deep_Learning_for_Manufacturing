% get local UCS to "partID" using PCA (Principal Component Analysis)
function Tucs=getPartUCS(data, partID)

idnodes=data.Model.Nominal.Domain(partID).Node;
node=data.Model.Nominal.xMesh.Node.CoordinateReset(idnodes,:);

% compute centre
Pc=mean(node,1);
    
% compute local principal reference system using PCA
node(:,1)=node(:,1)-Pc(1);
node(:,2)=node(:,2)-Pc(2);
node(:,3)=node(:,3)-Pc(3);
[V, ~] = eig(cov(node)); % the base of ghe decomposition gives the principal axis

% the normal vector is the eigenvector related to the smallest eigenvalue
Z=V(:,1); % => ...smallest eigenvalue
X=V(:,2);
Y=V(:,3);

% build rotation matrix
Rc=[X, Y, Z];

% check orthogonality
if det(Rc)<0
    Rc(:,2)=-Rc(:,2);
end

% save back
Tucs=eye(4,4);
Tucs(1:3,1:3)=Rc; Tucs(1:3,4)=Pc; 

