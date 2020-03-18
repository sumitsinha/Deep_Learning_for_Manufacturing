% calculate stochastic outputs (meand and std) using training data "Y"
function [Py, flag]=modelComputeStochasticOutput(data, Y, typeout)

% INPUT:
% data: data structure
% Y (nPC(nMC)*nD x nSd): sampled output values
% typeout: 1/2 => mean/standard deviation

% OUTPUT:
% Py: output (nD, nSd) => mean/standard deviation
    % nSd: no. stocastic dependent variables
    % nD: no. deterministic instances   
    
%----------
if data.Assembly.Solver.PolChaos.UsePolChaos % use polynomial chaos
    [Py, flag]=localUpdatePolChaos(data, Y, typeout);
else % use monte carlo
    [Py, flag]=localUpdateMC(data, Y, typeout);
end


%------ build using  pol chaos expansion
function [Py, flag]=localUpdatePolChaos(data, Y, typeout)

% Y (nPC*nD x nSd): sampled output values
% typeout: 1/2 => mean/standard deviation

%--------
nsample=data.Assembly.Solver.PolChaos.PopulationSize;    

% get counters
nSd=size(Y,2);
nD=data.Assembly.X.nD;
Py=zeros(nD, nSd);

% get PC coefficients
[alfa, flag]=solvePolChaosCoeff(data, Y);

if ~flag
    return
end

% get counters
degree=data.Assembly.Solver.PolChaos.Degree;
s=data.Assembly.Solver.PolChaos.RatioSample;
nPC=data.Assembly.PolChaos.nPC;
N=nPC/s;
nSi=data.Assembly.PolChaos.nSi;

%-----------------------------------------------
% get cdfs by sampling ("nsample" times)
cfi=getChaosPoly(degree, nsample, nSi, N);

for i=1:nD
     for j=1:nSd
         u=alfa(:,j,i)' * cfi;
         %--
         if typeout==1 % mean
             Py(i, j)=mean(u);
         elseif typeout==2 % standard deviation
             Py(i, j)=std(u);
         end
     end
end
   
%----------    
function cfi=getChaosPoly(degree, nsample, nSi, N)

% degree: chaos degree
% nsample: no. of sample for KDE evaluation
% N: no. of coefficients
% nSi: no. stocastic independent variables

% cfi: evaluated polynomial chaos expansion

cfi=zeros(N, nsample);
for k=1:nsample

    xin=randn(1,nSi);

    for j=1:N
        cfi(j, k)=hermiten(degree, nSi, j, xin);
    end

end

%--- Build using monte carlo
function [Py, flag]=localUpdateMC(data, Y, typeout)

% Y (nMC*nD x nSd): sampled output values
% typeout: 1/2 => mean/standard deviation

flag=true;

% get counters
nT=size(Y,1);
nSd=size(Y,2);
nD=data.Assembly.X.nD;
nMC=data.Assembly.Solver.PolChaos.MaxIter;
Py=zeros(nD, nSd);

% check
if nT~=nMC*nD
    flag=false;
    return
end

% run...
for i=1:nD
     for j=1:nSd
         ii=linspace(1, nMC, nMC);
         iii=(ii-1)*nD + i;
         u=Y(iii, j);
         %--
         if typeout==1 % mean
             Py(i, j)=mean(u);
         elseif typeout==2 % standard deviation
             Py(i, j)=std(u);
         end
     end
end
