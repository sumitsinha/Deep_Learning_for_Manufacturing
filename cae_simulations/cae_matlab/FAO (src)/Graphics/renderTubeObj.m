% plot 3D tube
function [X, Y, Z]=renderTubeObj(curve, rc)

% curve: (xyz) coordinate - [np, 3]
% rc: radius of tube

X=[];
Y=[];
Z=[];

%--
np=size(curve,1);

% loop over all segments
for i=1:np-1
    
    % point
    Pc=curve(i,:);
    
    % length
    ll=0;
    lu=norm(curve(i+1,:)-curve(i,:));
    
    % cylinder axis
    Nc=(curve(i+1,:)-curve(i,:))/lu;
    
    % create surface
    [Xi,Yi, Zi]=renderCylObj(rc, ll, lu, Nc, Pc);
   
    % stack-up pieces
    X=[X;Xi];   
    Y=[Y;Yi];
    Z=[Z;Zi];
   
end

