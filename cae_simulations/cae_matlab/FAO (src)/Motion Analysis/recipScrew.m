% Orthogonal to "S"
function Sr=recipScrew(S)
    % S: screw matrix
    % Sr: ricoprocal screw
    
% Solve null space "S*Sr=0"
tempSr=null(S)';
[nr,nc]=size(tempSr);

if nr>0
    Sr=zeros(nr,nc);

    % swap the first triplet with the last one
    Sr(:,1:3)=tempSr(:,4:6);
    Sr(:,4:6)=tempSr(:,1:3);

    %..
    Sr=rref(Sr);
else
    Sr=[];
end