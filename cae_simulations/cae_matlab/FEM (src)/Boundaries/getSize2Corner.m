% calculate additional points for each uni. constraint
function P=getSize2Corner(Nm, Pm, sizeFlag, Nt, A, B)

% Nm/Nt: normal and tanget dir
% Pm: center point
% A/B: sizes

if sizeFlag
    
    % get rotation matrix
    x=cross(Nt,Nm);
    x=x/norm(x);

    y=cross(Nm, x);

    R0c=[x', y', Nm'];

    % 4-point
    P=[A/2 B/2 0
       -A/2 B/2 0
       -A/2 -B/2 0
       A/2 -B/2 0];
   
    % transform in global frame
    P=apply4x4(P, R0c, Pm);

    % add Pm to the list
    P=[Pm;P];
    
else
    
    % 1-point
    P=Pm;
    
end

