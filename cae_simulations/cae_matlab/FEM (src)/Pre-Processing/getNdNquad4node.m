%- get shape functions and their derivatives for a 4-node element
function [N,dN]=getNdNquad4node(csi,eta)
% csi/eta: natural space

% node numbering
% 2-------------------1




% 3-------------------4

    N=zeros(1,4);
    dN=zeros(2,4);
    
    % shape functions
    N(1)=1/4*(1+csi)*(1+eta);
    N(2)=1/4*(1-csi)*(1+eta);
    N(3)=1/4*(1-csi)*(1-eta);
    N(4)=1/4*(1+csi)*(1-eta);
    
    % derivative over csi
    dN(1,1)=1/4*(1+eta);
    dN(1,2)=-1/4*(1+eta);
    dN(1,3)=-1/4*(1-eta);
    dN(1,4)=1/4*(1-eta);
    
    % derivative over eta
    dN(2,1)=1/4*(1+csi);
    dN(2,2)=1/4*(1-csi);
    dN(2,3)=-1/4*(1-csi);
    dN(2,4)=-1/4*(1+csi);
    
    