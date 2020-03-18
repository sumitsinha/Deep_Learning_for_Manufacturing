% perform 3D-DCT decomposition
function data=DCT3Decomposition(fem, cloud)

% fem: fem structure
% cloud, xyz coordinates of cloud of points

tic;

% get option
energylev=fem.Dct.Option.Energy;
corrlev=fem.Dct.Option.CorrThr;
weightcorr=fem.Dct.Option.WeightCorrection;

% STEP 1: get voxel deviations
data=DCT3Deviation2Voxel(fem, cloud);

% get voxel resolution
switch fem.Dct.Option.VoxelSelection
    
    case {1}
    %based on Voxel Size input 
    Nvox(1)=fem.Dct.Option.VoxX;
    Nvox(2)=fem.Dct.Option.VoxY;
    Nvox(3)=fem.Dct.Option.VoxZ;
    case{2}
    %based on percentage input of the voxel
    Nvox(1)= data.VoxX;
    Nvox(2)= data.VoxY;
    Nvox(3)= data.VoxZ;
    
end

% STEP 2: perform DCT
disp('DCT: Performing 3D DCT calculation');
data.Coeff = DCT3(data.VoxelDev);

% STEP 3: get energy compaction
disp('DCT: Sub-set #1 - Performing Energy Compaction');
data=DCT3EnergyCompaction(fem, data);

% STEP 4: get correlation calculation
disp('DCT: Sub-set #2 Getting Correlation value');
data = DCT3GetCorrelation(fem, data);

disp('DCT: Merged Sub-set = Sub-set #1 Merged with Sub-set #2');
data=DCT3CoefficientMerging(fem, data);

% STEP 5: perform least square approach
disp('DCT: Peforming weight correction-Least Square Approach');
A = data.NodeDevMerged'*data.NodeDevMerged;
b = data.NodeDevMerged'*data.NodeDevOriginal;

% get related weights
data.CorrCoeffWeight = A\b;

% STEP 6: create voxel structure for inverse DCT
disp('DCT: Creating Voxel Structure for inverse DCT ');

mode=zeros(Nvox(1),Nvox(2),Nvox(3));

for i=1:length(data.MergedCoeff)
    % get ids
    id=data.MergedLocationCoeff(i);

    % apply weight correction
    if weightcorr
        mode(id)=data.MergedCoeff(i) * data.CorrCoeffWeight(i);
    else
        mode(id)=data.MergedCoeff(i);
    end
    
    data.SelectedCoeffValue(i) = mode(id); %Save selected Coefficient value after weightage multipication from least square approach
    data.SelectedCoeffLocation(i) = id;
    
end 

data.NoSelectedCoeffLocation = length(data.SelectedCoeffLocation); 


% STEP 7: perform inverse DCT 
data.CoeffInv= DCT3Inv(mode);

tm=toc;

% summary:
fprintf('DCT - Summary:\n');
fprintf('       Percentage of energy compaction: %f\n',energylev);
fprintf('       No. of Coefficient (after energy compaction): %g\n',data.EnergyNcoeff);
if fem.Dct.Option.EnergyCoeffManual
    fprintf('       No. of Coefficient (after Manual Selection): %g\n',data.EnergyNcoeffManual);
end
fprintf('       Percentage of correlation threshold: %f\n',corrlev);
fprintf('       No. of Coefficient (after correlation truncation): %g\n',data.CorrNcoeff);
if fem.Dct.Option.CorrCoeffManual
    fprintf('       No. of Coefficient (after correlation truncation): %g\n',data.CorrNcoeffManual);
end
fprintf('       No. of Coefficient (selected for surface reconstruction): %g\n',data.NoSelectedCoeffLocation);

fprintf('       Computational time: %f sec.\n',tm);





