% Rodrigues rotation matrix
function R = RodriguesRot(V,angle)
% V = [1x3]
%angle = radians

if size(V,1)>1
    V=V';
end

% define axial matrix
W = zeros(3,3);
W(1,2) = -V(3);
W(1,3) = V(2);
W(2,1) = V(3);
W(2,3) = -V(1);
W(3,1) = -V(2);
W(3,2) = V(1);

% calculate rotation
K= V'*V;
R = K + cos(angle)*(eye(3,3) - K) + sin(angle)*W;


