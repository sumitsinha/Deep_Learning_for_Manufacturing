% count no. of parameters for stochastic simulation on parts
function [nSip, nSi]=countStochasticParametersPart(data, idparts, tparts)

% INPUT
% data: data structure
% idparts: list of parts
% tparts: type of parts (1='Deterministic'; 2='Stochastic')

% OUTPUT
% nSip: no. stochastic independent variables (per part) => (-1; 0; >1) => nominal; deterministic; stochastic
% nSi: no. of total stochastic independent variables. If nSi==0 then the solution is deterministic

% set initial ouputs
np=length(idparts);
nSip=zeros(1, np);
nSi=0; % deterministic by default

% loop over all parts
count=1;
for idpart=idparts
    
    gpart=data.Input.Part(idpart).Geometry.Type{1}; % part geometry
    
    if gpart==1 % nominal
       nSip(count)=-1; 
    elseif gpart==2  % morphed
    
        if tparts(count)==2 % stochastic
            r=0;
            if ~isempty(data.Input.Part(idpart).Morphing)
                r=length(data.Input.Part(idpart).Morphing);
            end
            for k=1:r
                distrub=data.Input.Part(idpart).Morphing(k).Distribution{1};

                if distrub>1 % stochastic               
                   nSip(count)=nSip(count)+1;
                   nSi=nSi+1;
                end
            end

        else % determistic
            nSip(count)=0;
        end
        
    elseif gpart==3  % measured
        nSip(count)=0; 
    end
    
    count=count+1;

end