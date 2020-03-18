% This function gives the min angle on a triangle in 3D

function angle=getMinAngleTria3D(P)

% P: triangle coordinate


% first angle
v1=(P(2,:)-P(1,:))/norm(P(2,:)-P(1,:));
v2=(P(3,:)-P(1,:))/norm(P(3,:)-P(1,:));
angle(1)=acos(dot(v1,v2))*180/pi;

% second angle
v1=(P(1,:)-P(3,:))/norm(P(1,:)-P(3,:));
v2=(P(2,:)-P(3,:))/norm(P(2,:)-P(3,:));
angle(2)=acos(dot(v1,v2))*180/pi;

% third angle
v1=(P(1,:)-P(2,:))/norm(P(1,:)-P(2,:));
v2=(P(3,:)-P(2,:))/norm(P(3,:)-P(2,:));
angle(3)=acos(dot(v1,v2))*180/pi;

%--
angle=min(angle);


