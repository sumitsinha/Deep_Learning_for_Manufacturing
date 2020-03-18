% get non-zero coefficient in the voxel structure
function data = DCT3GetCoeffLocation(fem, data) 

% fem: fem structure
% data, dct data structure

% get inputs
Nvox(1)=fem.Dct.Option.VoxX;
Nvox(2)=fem.Dct.Option.VoxY;
Nvox(3)=fem.Dct.Option.VoxZ;

% set initial values
data.EnergyLocationCoeff =[];
data.EnergyCoeff = [];

count=1;
for k=1:Nvox(3)
    
    for j=1:Nvox(2)
        
        for i=1:Nvox(1)
            
            if data.Coeff(i,j,k) ~= 0 % if the coeff is not zero
                
                % location and coefficient
                data.EnergyLocationCoeff(count,:) = [i, j, k]; 
                data.EnergyCoeff(count) = data.Coeff(i,j,k);
                count = count+1;
                
            end
            
        end
        
    end
end
