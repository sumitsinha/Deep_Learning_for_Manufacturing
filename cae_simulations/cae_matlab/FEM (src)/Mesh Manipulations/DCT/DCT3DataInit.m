% set initial structure
function data=DCT3DataInit()

% BB size and location
data.BBox.deltax=[];
data.BBox.deltay=[];
data.BBox.deltaz=[];

data.BBox.X=[];
data.BBox.Y=[];
data.BBox.Z=[];

data.BBox.minx=[];
data.BBox.miny=[];
data.BBox.minz=[];

% voxel ids and related deviations
data.VoxelSize=[]; % storing the voxel size
data.VoxelId=[]; % voxel structure
data.NoEmpVoxelId=[]; % not empty voxel
% data.EmpVoxelId=[]; % empty voxel

data.VoxelDev=[]; % voxel deviations

% SET 1: overall coefficients data set
data.Coeff=[]; % coefficient after DCT decomposition (voxel structure)
data.CoeffInv=[]; % deviations - inverse DCT (voxel structure)

% SET 2: energy compaction
data.EnergyLocationCoeff=[]; % coefficient location [1, m]
data.EnergyCoeff=[]; % coefficient value [1, m]
data.EnergyNcoeff=0; % no. of coefficient: m
data.EnergyNcoeffManual=0; %No. of Coeffificents based on Manual Selection
data.TotalEnergy = 0; % Total Energy of the part

% SET 3: correlated coefficient
data.Corr=[]; % correlation values [1, n]
data.CorrLocationCoeff=[]; % location of correlated coefficient [1, n]
data.CorrCoeff=[]; % coefficient value [1, n]
data.CorrNcoeff=0; % no of correlated coefficient: n
data.CorrNcoeffManual=0; %No. of Coeffificents based on Manual Selection

% SET 4: merged set
data.MergedLocationCoeff=[]; 
data.MergedCoeff=[]; 
data.LinearIndexEnergy=[]; % used to move from global data set to local data sets
data.LinearIndexCorr=[];

% SET 5: weight coefficients for least square
data.CorrCoeffWeight=[]; % related weight

% geometry reconstruction
data.NodeDevOriginal=[]; % original deviations
data.NodeDev=[]; % reconstructed deviations
data.NodeDevMerged=[]; % node deviations based on correlation

data.NodeDevEnergy=[]; % deviations related to coefficient after energy compaction

data.SelectedCoeffValue=[];% selection of coefficient based on selected number input
data.NoSelectedCoeffLocation = []; % No of coefficient selected for surface reconstruction
data.SelectedCoeffLocation = []; % Coefficient value after least square approach (weighted coefficient) 



