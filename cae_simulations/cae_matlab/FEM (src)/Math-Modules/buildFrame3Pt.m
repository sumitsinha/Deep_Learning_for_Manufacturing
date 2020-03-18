function [R, d]=buildFrame3Pt(P0, P1, P2)

% get x
X=(P1-P0)/norm(P1-P0);

% get tempY
ty=(P2-P0)/norm(P2-P0);

% get Z
Z=cross(X, ty)/norm(cross(X, ty));

% get Y
Y=cross(Z, X);

% rotation matrix
R=[X' Y' Z'];

% origin
d=P0;