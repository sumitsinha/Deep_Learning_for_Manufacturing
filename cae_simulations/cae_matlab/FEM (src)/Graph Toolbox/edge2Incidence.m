%- calculate incidence matrix from edge matrix
function I=edge2Incidence(E)

nE=size(E,1);
nV=max(E(:));

I=zeros(nV,nE);

for i=1:nE
    I(E(i,1),i)=1;
    I(E(i,2),i)=-1;
end

