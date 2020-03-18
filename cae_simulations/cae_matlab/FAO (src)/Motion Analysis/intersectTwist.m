% Interection of set of twist
function TR=intersectTwist(T1, T2)
    % T1/T2: input twist matrices
    % TR: resultant twist matrix
    
% Orthogonal 
rT1=recipScrew(T1);
rT2=recipScrew(T2);

% Union
rT=[rT1;rT2];

% Orthogonal to rT
TR=recipScrew(rT); 
