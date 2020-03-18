% trasforma inversa del punto "P"
function P=applyinv4x4(P, R, d)

d=-R'*d';
R=R';

P=(R*P')';
P(:,1)=P(:,1)+d(1);
P(:,2)=P(:,2)+d(2);
P(:,3)=P(:,3)+d(3);