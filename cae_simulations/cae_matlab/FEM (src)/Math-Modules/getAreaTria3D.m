% This function gives the area of a triangle in 3D

function area=getAreaTria3D(P)

% P: triangle coordinate

% area=sqrt(s*(s-a)*(s-b)*(s-c)) - Heron's formula

% area=1/4*sqrt( (a+b+c)*(c-a+b)*(c+a+b)*(a+b-c) ) stable formula

a=norm(P(2,:)-P(1,:));
b=norm(P(3,:)-P(1,:));
c=norm(P(3,:)-P(2,:));

L=sort([a, b, c]);
a=L(1); b=L(2); c=L(3);

area=1/4*sqrt( (a+b+c)*(c-a+b)*(c+a-b)*(a+b-c) );



