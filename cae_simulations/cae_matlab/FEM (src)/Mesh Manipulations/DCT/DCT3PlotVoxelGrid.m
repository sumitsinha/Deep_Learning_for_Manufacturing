% plot the voxel grid
function DCT3PlotVoxelGrid(fem, data)

% get inputs
M=fem.Dct.Option.VoxX;
N=fem.Dct.Option.VoxY;
T=fem.Dct.Option.VoxZ;

xlim=data.BBox.X;
ylim=data.BBox.Y;
zlim=data.BBox.Z;

stepx=data.BBox.deltax/M;
stepy=data.BBox.deltay/N;
stepz=data.BBox.deltaz/T;

% loop over all points
%--  
point=[];
face=[];
count=-8;

pointn=[];
facen=[];
countn=-8;

nemp=size(data.EmpVoxelId,1); % empty voxels
nnemp=size(data.NoEmpVoxelId,1); % not empty voxels

% loop over empty voxels
for i=1:nemp
    
     ii=data.EmpVoxelId(i,1);
     jj=data.EmpVoxelId(i,2);
     kk=data.EmpVoxelId(i,3);
     
     [pointt, facet, countn]=renderVoxel(stepx, stepy, stepz,...
                                        ii, jj, kk, countn);

     pointt(:,1)=pointt(:,1)+xlim(2);
     pointt(:,2)=pointt(:,2)+ylim(2);
     pointt(:,3)=pointt(:,3)+zlim(2);

     facen=[facen
           facet];

     pointn=[pointn
            pointt];
                    
end

% loop over empty voxels
for i=1:nnemp
    
     ii=data.NoEmpVoxelId(i,1);
     jj=data.NoEmpVoxelId(i,2);
     kk=data.NoEmpVoxelId(i,3);
     
     [pointt, facet, count]=renderVoxel(stepx, stepy, stepz,...
                                       ii, jj, kk, count);

     pointt(:,1)=pointt(:,1)+xlim(2);
     pointt(:,2)=pointt(:,2)+ylim(2);
     pointt(:,3)=pointt(:,3)+zlim(2);

     face=[face
           facet];

     point=[point
            pointt];
                    
     
end

% plot data
point=point(:,[2 1 3]);
point(:,2)=-point(:,2);

% plot sections     
patch('faces',face,...
       'vertices',point,...
       'edgecolor','k',...
       'facecolor','r',...
        'parent',fem.Post.Options.ParentAxes)

pointn=pointn(:,[2 1 3]);
pointn(:,2)=-pointn(:,2);

patch('faces',facen,...
         'vertices',pointn,...
         'edgecolor','k',...
         'facecolor','none',...
         'parent',fem.Post.Options.ParentAxes)
   
%
view(3)
axis equal

if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end


%- render voxel
function [pointvox, face, count]=renderVoxel(stepx, stepy, stepz,...
                                      i, j, k, count)
                                                
pointvox=[stepx*(i-1)  stepy*(j-1) stepz*(k-1)
          stepx*(i-1)+stepx  stepy*(j-1) stepz*(k-1)
          stepx*(i-1)+stepx  stepy*(j-1)+stepy stepz*(k-1)
          stepx*(i-1) stepy*(j-1)+stepy stepz*(k-1)
          stepx*(i-1)  stepy*(j-1) stepz*(k-1)+stepz
          stepx*(i-1)+stepx  stepy*(j-1) stepz*(k-1)+stepz
          stepx*(i-1)+stepx  stepy*(j-1)+stepy stepz*(k-1)+stepz
          stepx*(i-1) stepy*(j-1)+stepy stepz*(k-1)+stepz];
      
face=[1 2 3 4
      5 6 7 8
      1 2 6 5
      4 3 7 8
      2 6 7 3
      1 5 8 4];

count=count+8; 
face=face+count;



