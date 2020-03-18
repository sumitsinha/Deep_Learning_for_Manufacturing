% Reset solution
function data=modelReset(data, flagp, flaga)

% Input:
% data: input model
% flag
    % flagp => if true: reset part data 
    % flaga => if true: reset assembly data
  
% Input:
% data: updated input model
%---------

if nargin==1
    flagp=false; 
    flaga=false; 
end

if nargin==2
    flaga=false; 
end

%--
if flaga
    data.Assembly.PolChaos.A=[];
    data.Assembly.PolChaos.Csi=[];
    data.Assembly.PolChaos.nPC=1; 
    data.Assembly.PolChaos.nSip=[];
    data.Assembly.PolChaos.nSi=0;
    %
    data.Assembly.U=[];
    data.Assembly.U{1}=[];
    %
    data.Assembly.GAP=[];
    data.Assembly.GAP{1}=[];
    %
    data.Assembly.Log=[];
    data.Assembly.Log{1}=[];
end
%--
for i=1:length(data.Input.Part)
    
    if flaga  % reset assembly data
        data.Input.Part(i).U=[];
        data.Input.Part(i).U{1}=[];
    end
    
    if flagp % reset part data
        data.Input.Part(i).D=[];
        data.Input.Part(i).D{1}=[];
    end
end
%