function N=getNormalCoP(Point, dsearch)

% INPUT:
% Point: xyz coordinates of the CoP
% dsearch: searching distance

% OUTPUT
% N: normal vectors 

%------------
% Use the covariance matrix to detect the normal vector (eigenvector related to the smallest eigenvalue) of a given point
%------------

% define view vector for normal direction repairing
Vp=[1 0 0];
%-----------------------------------------------

% no. of points
np=size(Point,1);

% set initial values
N=zeros(np,3);

% loop over all points
for i=1:np
   
    Pi=Point(i,:);
    
    % Point-Pi
    temp(:,1)=Point(:,1)-Pi(1);
    temp(:,2)=Point(:,2)-Pi(2);
    temp(:,3)=Point(:,3)-Pi(3);
    
    % get distances
    d=sqrt(sum(temp.^2,2));
    
    % so closest points are
    Ps=Point(d<=dsearch,:);
    
    % apply PCA on those points
    [V, ~] = eig(cov(Ps));
    
    % the normal vector is the eigenvector related to the smallest eigenvalue
    Ni=V(:,1);
    
    % check for consistency of normal (based on: http://pointclouds.org/documentation/tutorials/normal_estimation.php)
    S=(Vp-Pi);
    if dot(Ni,S)<=0;
        Ni=-Ni;
    end
   
    % store
    N(i,:)= Ni;
    
end

