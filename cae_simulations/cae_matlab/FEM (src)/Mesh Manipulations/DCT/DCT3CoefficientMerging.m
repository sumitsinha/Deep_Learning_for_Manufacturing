% merge coefficient data set based on energy compaction and correlation
function data=DCT3CoefficientMerging(fem, data)

% set intial values
data.MergedLocationCoeff=[]; 
data.MergedCoeff=[]; 

% check manual selection
if fem.Dct.Option.EnergyCoeffManual || fem.Dct.Option.CorrCoeffManual
    
if fem.Dct.Option.EnergyCoeffManual
    
%     data.EnergyLocationCoeffManual=[];
    noEnergyCoeff = fem.Dct.Option.NoEnergyCoeff;
    
    if noEnergyCoeff <= data.EnergyNcoeff
        disp('... using manual selection for energy compaction')
    else
        noEnergyCoeff=data.EnergyNcoeff;
        disp('... using automatic selection for energy compaction')
    end
        
    data.EnergyLocationCoeff = data.EnergyLocationCoeff(1:noEnergyCoeff);
    data.EnergyCoeff = data.EnergyCoeff(1:noEnergyCoeff);
    
    data.LinearIndexEnergy=data.LinearIndexEnergy(1:noEnergyCoeff);
    
    data.EnergyNcoeffManual=noEnergyCoeff;
    
end

%-- selection of correlation coefficient based on manual input
if fem.Dct.Option.CorrCoeffManual
    
    noCorrCoeff = fem.Dct.Option.NoCorrCoeff;
    
    % check if the manual input is consistent
    if noCorrCoeff <= length(data.CorrLocationCoeff)
        
        disp('... using manual selection for correlation')
    else
        noCorrCoeff=data.CorrNcoeff;
        disp('... using automatic selection for correlation')
    end
    
    data.CorrLocationCoeff = data.CorrLocationCoeff(1:noCorrCoeff);
    data.CorrCoeff = data.CorrCoeff(1:noCorrCoeff);
    
    data.LinearIndexCorr=data.LinearIndexCorr(1:noCorrCoeff);
    
    data.CorrNcoeffManual=noCorrCoeff;

end

% save location
% (ENERGY)
data.MergedLocationCoeff=[data.MergedLocationCoeff
                          data.EnergyLocationCoeff];
% (CORRELATION)
data.MergedLocationCoeff=[data.MergedLocationCoeff
                          data.CorrLocationCoeff];
             
% save coefficients value
% (ENERGY)
data.MergedCoeff=[data.MergedCoeff
                 data.EnergyCoeff];
% (CORRELATION)
data.MergedCoeff=[data.MergedCoeff
                  data.CorrCoeff];
              
% save linear index
LinearIndex = [data.LinearIndexEnergy data.LinearIndexCorr];
             
% Get the unique coefficient locationwise        
temp = cat(2,data.MergedLocationCoeff,data.MergedCoeff,LinearIndex');
temp = unique(temp,'rows');

data.MergedLocationCoeff =temp(:,1);    % save unique coefficient location
data.MergedCoeff = temp(:,2);           % save unique coefficient value
index=temp(:,3);                        % save linear index
% save node deviations
data.NodeDevMerged=data.NodeDevEnergy(:, index');


else
%% only to select coefficients based on the correlation
data.MergedLocationCoeff=[data.MergedLocationCoeff
                          data.CorrLocationCoeff];
             
% save coefficients value
% (CORRELATION)
data.MergedCoeff=[data.MergedCoeff
                  data.CorrCoeff];
              
% save linear index
LinearIndex = data.LinearIndexCorr;
             
% Get the unique coefficient locationwise        
temp = cat(2,data.MergedLocationCoeff,data.MergedCoeff,LinearIndex');
temp = unique(temp,'rows');

data.MergedLocationCoeff =temp(:,1);    % save unique coefficient location
data.MergedCoeff = temp(:,2);           % save unique coefficient value
index=temp(:,3);                        % save linear index
% save node deviations
data.NodeDevMerged=data.NodeDevEnergy(:, index');

end

