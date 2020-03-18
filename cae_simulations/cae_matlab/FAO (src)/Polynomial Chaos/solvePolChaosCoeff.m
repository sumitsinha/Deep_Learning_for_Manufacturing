% calculate coefficient of the polynomial chaos (PC) using least square method
function [alfa, flag]=solvePolChaosCoeff(data, Y)

% INPUT:
% data: data structure
% Y (nPC*nD x nSd): sampled output values

% OUTPUT:
% alfa(nPC x nSd x nD): coefficients of chaos expansion
    % nPC: no. of pol chaos coefficients
    % nSd: no. stocastic dependent variables
    % nD: no. deterministic instances

%------------------------------
alfa=[];
flag=true;

A=data.Assembly.PolChaos.A;

%--
nSd=size(Y,2);
nPC=data.Assembly.PolChaos.nPC;
nD=data.Assembly.X.nD;
nT=size(Y,1);
%--

if nT~=nPC*nD
    flag=false;
    return
end

% allocate load matrix
b=zeros(nPC, nSd, nD);

for i=1:nT
    ii=ceil(i/nD);
    iii=i-nD*(ii-1);
    
    b(ii,:,iii)=Y(i,:);
end
    
% solve least squares problem and get coefficients
s=data.Assembly.Solver.PolChaos.RatioSample;
N=nPC/s;
alfa=zeros(N, nSd, nD);

for i=1:nD
     alfa(:, :, i)=A\b(:, :, i);
end
