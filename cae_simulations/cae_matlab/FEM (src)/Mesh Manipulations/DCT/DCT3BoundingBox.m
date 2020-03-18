% get Bounding box in image coordinate frame
function data=DCT3BoundingBox(fem)

% fem: fem structure

% data:
      % SIZE
%     data.BBox.deltax=maxx-minx;
%     data.BBox.deltay=maxy-miny;
%     data.BBox.deltaz=maxz-minz;
%
      % Extreme points of BB
%     data.BBox.X=[maxx minx];
%     data.BBox.Y=[maxy miny];
%     data.BBox.Z=[maxz minz];

% initialize data structure
data=DCT3DataInit;

% get nodes
idpart=fem.Dct.Domain;

% get scaling factor for bb
pr=fem.Dct.Option.ScaleBB;

% percentage for no. of voxels
vpr=fem.Dct.Option.VoxelPercentage;

%-----
idnode=fem.Domain(idpart).Node;

point=fem.xMesh.Node.Coordinate(idnode,:);

% use image coordinate frame (I am flipping "y" coordinate)
point(:,2)=-point(:,2);
point=point(:,[2 1 3]);

% get bounding box around the point cloud
maxx=max(point(:,1));
minx=min(point(:,1));

maxy=max(point(:,2));
miny=min(point(:,2));

maxz=max(point(:,3));
minz=min(point(:,3));

% save for back-up
data.BBox.minx=minx;
data.BBox.miny=miny;
data.BBox.minz=minz;

%-----------------------
deltax=maxx-minx;
deltay=maxy-miny;
deltaz=maxz-minz;

% scale factor to avoid degenerated b. box
sc=pr*max([deltax deltay deltaz]);

% update all
maxx=maxx+sc/2;
maxy=maxy+sc/2;
maxz=maxz+sc/2;

minx=minx-sc/2;
miny=miny-sc/2;
minz=minz-sc/2;

% get delta along axes and save all
data.BBox.deltax=maxx-minx;
data.BBox.deltay=maxy-miny;
data.BBox.deltaz=maxz-minz;

% get Voxel Size
data.VoxX = fix(vpr*data.BBox.deltax);
data.VoxY = fix(vpr*data.BBox.deltay);
data.VoxZ = fix(vpr*data.BBox.deltaz);

%--
data.BBox.X=[maxx minx];
data.BBox.Y=[maxy miny];
data.BBox.Z=[maxz minz];


