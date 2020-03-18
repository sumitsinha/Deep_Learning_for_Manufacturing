%- calculate jacobian matrix for 2D element
function [J, detJ]=getJacobian2D(dN, P)
  
  J=zeros(2,2);
  
  %- node coordinates
  x=P(:,1);
  y=P(:,2);
 
  n=length(x);
  
  % calculate entries
  J(1,1)=dot(dN(1,1:n),x); 
  J(1,2)=dot(dN(1,1:n),y);
  
  J(2,1)=dot(dN(2,1:n),x);
  J(2,2)=dot(dN(2,1:n),y);
 
  
  % and the related determinat
  detJ=det(J);
  
  