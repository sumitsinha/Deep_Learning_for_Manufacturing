% Interection of set of wrench
function WR=intersectWrench(T1, T2)
    % T1/T2: input twist matrices
    % WR: resultant twist matrix
       
% Union
rW=[T1;T2];

% Orthogonal to rW
WR=recipScrew(rW); 
