%- get shape functions and their derivatives for a 3-node element
function [N,dN]=getNdNtria3node(csi,eta)
% csi/eta: natural space

% node numbering
% 3




% 1-------------------2

    N=zeros(1,3);
    dN=zeros(2,3);
    
    % shape functions
    N(1)=1-csi-eta;
    N(2)=csi;
    N(3)=eta;
    
    % derivative over csi
    dN(1,1)=-1;
    dN(1,2)=1;
    dN(1,3)=0;
    
    % derivative over eta
    dN(2,1)=-1;
    dN(2,2)=0;
    dN(2,3)=1;
    
    