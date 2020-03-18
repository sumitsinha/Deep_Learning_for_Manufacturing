% get DCT coefficient for a given level of energy compaction
function data=DCT3EnergyCompaction(fem, data)

% fem: fem structure
% data: dct data structure

% energy level
energylev=fem.Dct.Option.Energy;

% set initial values
data.EnergyLocationCoeff =[];
data.EnergyCoeff = [];
data.EnergyNcoeff=0;

% sort
[coeff, ind] = sort(abs(data.Coeff(:)),'descend'); % sort coefficient

% cumulative norm
cs=cumulativenorm(coeff);

% Compute total energy of the part 
data.TotalEnergy = cs(end);

% normalise
cs=cs/cs(end);

% select
data.EnergyLocationCoeff=ind(cs<=energylev);

coeff=data.Coeff(ind);
data.EnergyCoeff=coeff(cs<=energylev);

data.EnergyNcoeff=length(data.EnergyLocationCoeff);


%--
% function B=cumulativenorm(A)
% 
% n=length(A);
% 
% B=zeros(length(A),1);
% 
% for i=1:n
%     temp=zeros(1,n);
%     temp(1:i)=A(1:i);
%     
%     B(i)=norm(temp);
% end


