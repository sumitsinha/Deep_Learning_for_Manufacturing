% Find correlation between each energy compacted coefficient and original data
function  data = DCT3GetCorrelation(fem, data)

% fem: fem structure
% data, dct data structure

% get inputs
CorrThr=fem.Dct.Option.CorrThr;

data.Corr = zeros(data.EnergyNcoeff,1);
data.NodeDevEnergyOverallLocation=zeros(1,data.EnergyNcoeff);

% get linear index in the local energy data set
data.LinearIndexEnergy=1:data.EnergyNcoeff;

% correlation calculation
for i=1:data.EnergyNcoeff
    
    modeCoeff = zeros(size(data.Coeff));
    
    % get voxel location
    ii=data.EnergyLocationCoeff(i);
    
    modeCoeff(ii) = data.Coeff(ii);
    
    data.CoeffInv= DCT3Inv(modeCoeff);
    
    [~, data]=DCT3CreateGeometry(fem, data);
        
    data.Corr(i)= corr(data.NodeDevOriginal,data.NodeDev);
    
    % save deviations
    data.NodeDevEnergy(:,i)=data.NodeDev;
        
end

% keep only the most correlated coefficient
[data.Corr,idCorr] = sort(data.Corr,'descend');

data.CorrLocationCoeff=data.EnergyLocationCoeff(idCorr);
data.CorrCoeff=data.EnergyCoeff(idCorr);
data.NodeDevCorr=data.NodeDevEnergy(:,idCorr);

% KEEP all coefficients bigger than the threshold limit
data.CorrCoeff(data.Corr<=CorrThr)=[];
data.CorrLocationCoeff(data.Corr<=CorrThr)=[];
data.CorrNcoeff=length(data.CorrCoeff);

data.NodeDevCorr(:,data.Corr<=CorrThr)=[];

% save linear index
data.LinearIndexCorr=1:data.EnergyNcoeff;

data.LinearIndexCorr=data.LinearIndexCorr(idCorr);

data.LinearIndexCorr=data.LinearIndexCorr(data.Corr> CorrThr);





