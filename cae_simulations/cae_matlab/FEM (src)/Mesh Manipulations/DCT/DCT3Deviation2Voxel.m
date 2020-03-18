% calculate voxelization based on point cloud (mesh nodes)
function data=DCT3Deviation2Voxel(fem, cloud)

% N.B.: if the point i-th belong to the voxel in location (idx, idy, idz) then store in there the deviation of point i-th

% fem: fem structure
% cloud, xyz coordinates of cloud of points

% data
%   data.VoxelDev: deviations for each voxel

% STEP 1
% run BB calculation
fprintf('DCT: Computing Bounding Box\n');

laplaceint=fem.Dct.Option.LaplaceInterp;

data=DCT3BoundingBox(fem);

% STEP 2: get deviations
% get nodes and options

% get voxel resolution
switch fem.Dct.Option.VoxelSelection
    
    case true
    %based on Voxel Size input 
    Nvox(1)=fem.Dct.Option.VoxX;
    Nvox(2)=fem.Dct.Option.VoxY;
    Nvox(3)=fem.Dct.Option.VoxZ;
    case false
    %based on percentage input of the voxel
    Nvox(1)= data.VoxX;
    Nvox(2)= data.VoxY;
    Nvox(3)= data.VoxZ;
end

data.VoxelSize=Nvox; % saving voxel size in data structure
fprintf('DCT: Voxel Size:  %d   %d   %d\n',data.VoxelSize(1),data.VoxelSize(2),data.VoxelSize(3));

% set initial value
count=ones(Nvox(1),Nvox(2),Nvox(3)); % store no. of points for  each voxel

data.VoxelId=cell(Nvox(1),Nvox(2),Nvox(3)); 
%--------------------------------------------
data.VoxelDev=zeros(Nvox(1),Nvox(2),Nvox(3));

fprintf('DCT: Computing deviations\n');
idpart=fem.Dct.Domain;

sdistance=fem.Dct.Option.SearchDistN;
rdistance=fem.Dct.Option.SearchDistR;

%-----
idnode=fem.Domain(idpart).Node;

point=fem.xMesh.Node.Coordinate(idnode,:);
normal=fem.xMesh.Node.Normal(idnode,:);

offset=fem.Dct.Option.Offset;
dev=getNormalDevPoints2Points(point, normal, cloud, sdistance, rdistance)-offset;

data.NodeDevOriginal=zeros(length(idnode),1);
data.NodeDevOriginal(idnode)=dev;

% STEP 3
% map from deviation to voxel space
fprintf('DCT: Computing Voxel deviations\n');

% get constants for MAPPING from coordinates space to voxel space
ax=(Nvox(1)-1)/data.BBox.deltax;
ay=(Nvox(2)-1)/data.BBox.deltay;
az=(Nvox(3)-1)/data.BBox.deltaz;

bx=1-ax*data.BBox.minx;
by=1-ay*data.BBox.miny;
bz=1-az*data.BBox.minz;

% use image coordinate frame
point(:,2)=-point(:,2);
point=point(:,[2 1 3]);

% loop over all points
np=size(point,1);

for i=1:np
    
    % get voxel indices
    idx=round(ax*point(i,1)+bx);
    idy=round(ay*point(i,2)+by);
    idz=round(az*point(i,3)+bz);
    
    % store all
    data.VoxelDev(idx,idy,idz)=data.VoxelDev(idx,idy,idz)+dev(i);

    data.VoxelId{idx,idy,idz}(end+1)=idnode(i);
    
    % save ids of not empty voxels
    data.NoEmpVoxelId(i,:)=[idx idy idz];
    
    count(idx,idy,idz)=length(data.VoxelId{idx,idy,idz});   
    
end

% remove duplicates and save it
data.NoEmpVoxelId=unique(data.NoEmpVoxelId,'rows');


% get average on each voxel
data.VoxelDev=data.VoxelDev./count; % devision element by element

% STEP 4
if laplaceint
    % perform laplace interpolation
    fprintf('DCT: Performing Laplace interpolation\n');

    dx=data.BBox.deltax/Nvox(1);
    dy=data.BBox.deltay/Nvox(2);
    dz=data.BBox.deltaz/Nvox(3);
    
    data.VoxelDev=laplace3D(data.VoxelDev, dx, dy, dz);
else
   fprintf('DCT: Laplace interpolation not performed\n'); 
end


