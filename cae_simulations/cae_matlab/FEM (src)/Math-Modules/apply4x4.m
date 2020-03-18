% trasforma il punto "P" con la rotazione "R" e la posizione "d"
function P=apply4x4(P, R, d)

P=(R*P')';
P(:,1)=P(:,1)+d(1);
P(:,2)=P(:,2)+d(2);
P(:,3)=P(:,3)+d(3);
