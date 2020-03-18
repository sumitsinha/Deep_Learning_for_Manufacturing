% get the normal vector of the plane passing through P1, P2, p3
function Nplane=point2Plane(P0, P1, P2)

Nplane=[0 0 0];

% get x
X=(P1-P0)/norm(P1-P0);

% get tempY
ty=(P2-P0)/norm(P2-P0);

% get Z
l=norm(cross(X, ty));
if l~=0
    Nplane=cross(X, ty)/l;    
end