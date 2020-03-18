% calculate pol chaos expansion
function [data, flag]=generatePolChaosExpasion(data, idparts, tparts)

% INPUT
% data: data structure
% idparts: list of parts
% tparts: type of parts (1='Deterministic'; 2='Stochastic')

% OUTPUT
% data: data structure
% flag: false/true => deterministic/stochastic solution

flag=true;

fprintf('Counting stochastic parameters...\n');

% count no. of independent stochastic variables (per part)
[nSip, nSi]=countStochasticParametersPart(data, idparts, tparts);

fprintf('      No. of stochastic parameters: %g\n', nSi);

if nSi==0
    
    % save out
    data.Assembly.PolChaos.A=[];
    data.Assembly.PolChaos.Csi=[];
    data.Assembly.PolChaos.nPC=1;
    data.Assembly.PolChaos.nSip=[];
    data.Assembly.PolChaos.nSi=0;
    
    flag=false;
    return
end

% run calculation
if data.Assembly.Solver.PolChaos.UsePolChaos % use polynomial CHAOS
    data=localUpdatePolChaos(data, idparts, nSip, nSi);
else % use MONTE CARLO
    data=localUpdateMC(data, idparts, nSip, nSi);
end

%--------------------------------------------------------------------------

%--------------------------------------------
function data=localUpdatePolChaos(data, idparts, nSip, nSi)

% read pol chaos degree and rate of sampling
degree=data.Assembly.Solver.PolChaos.Degree;
s=data.Assembly.Solver.PolChaos.RatioSample;

% no. of terms of chaos expansion
N=polChaosNoTerms(degree, nSi);

fprintf('      Minimum no. of polynomial chaos evaluations: %g\n', N);

% no. of random samples
nPC=N*s; 

fprintf('Calculating polynomial chaos expansion...\n');

A=zeros(nPC, N);
Csi=cell(nPC, length(idparts));
for i=1:nPC
    
    fprintf('      Chaos evaluation ID: %g\n', i);
        
    % GAUSSIAN random sample
    xin=[];
    c=1;
    for idpart=idparts
        
         if nSip(c)>0 % stochastic part

             r=length(data.Input.Part(idpart).Morphing);

             % get deviation at control points
             Csip=[];
             for kr=1:r
                distrub=data.Input.Part(idpart).Morphing(kr).Distribution{1};

                if distrub==2 % gaussian
                    mu=data.Input.Part(idpart).Morphing(kr).Parameter(1);
                    sigma=data.Input.Part(idpart).Morphing(kr).Parameter(2);
                    
                    xinj=randn(1);
                    xin=[xin xinj];
                    
                    Csip=[Csip, xinj*sigma+mu];
                else


                    % ADD HERE ANY OTHER pdf

                end

             end
             
             Csi{i, c}=Csip;
             
         end
         
         c=c+1;
    end
        
    % weight matrix
    for j=1:N
        A(i, j)=hermiten(degree, nSi, j, xin);
    end
    
end

% save out
data.Assembly.PolChaos.A=A;
data.Assembly.PolChaos.Csi=Csi;
data.Assembly.PolChaos.nPC=nPC;
data.Assembly.PolChaos.nSip=nSip;
data.Assembly.PolChaos.nSi=nSi;


%--------------------------------------------
function data=localUpdateMC(data, idparts, nSip, nSi)

% read pol chaos degree and rate of sampling
nMC=data.Assembly.Solver.PolChaos.MaxIter;

fprintf('Calculating Monte Carlo...\n');

Csi=cell(nMC, length(idparts));
for i=1:nMC
            
    % GAUSSIAN random sample
    xin=[];
    c=1;
    for idpart=idparts
        
         if nSip(c)>0 % stochastic part

             r=length(data.Input.Part(idpart).Morphing);

             % get deviation at control points
             Csip=[];
             for kr=1:r
                distrub=data.Input.Part(idpart).Morphing(kr).Distribution{1};

                if distrub==2 % gaussian
                    mu=data.Input.Part(idpart).Morphing(kr).Parameter(1);
                    sigma=data.Input.Part(idpart).Morphing(kr).Parameter(2);
                    
                    xinj=randn(1);
                    xin=[xin xinj];
                    
                    Csip=[Csip, xinj*sigma+mu];
                else


                    % ADD HERE ANY OTHER pdf

                end

             end
             
             Csi{i, c}=Csip;
             
         end
         c=c+1;
    end
            
end

% save out
data.Assembly.PolChaos.Csi=Csi;
data.Assembly.PolChaos.nSip=nSip;
data.Assembly.PolChaos.nSi=nSi;

