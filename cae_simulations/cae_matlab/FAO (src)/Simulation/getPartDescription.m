function [pdata, flag]=getPartDescription(data)

% data: model

% pdata: part data {part ID, geometry, simulation mode, parameter ID}
    % part ID: part ID
    % geometry: 1="nominal"; 2="morphed"; 3="measured"
    % simulation mode: 1="deterministic"; 2="stochastic"
    % parameter ID: parameter ID used to generate "measured" part
% flag: false=> no part active

%--
pdata=[];
flag=true;

%--
if ~isfield(data.Input,'Part')
    flag=false;
    fprintf('Getting part description(warning): no active !\n');
    return
end
npart=length(data.Input.Part);

c=0;
for i=1:npart
    if data.Input.Part(i).Status==0
       c=c+1;
       pdata(c,1)=i; 

       ge=data.Input.Part(i).Geometry.Type{1};
       pdata(c,2)=ge;

       ge=data.Input.Part(i).Geometry.Mode{1};
       pdata(c,3)=ge;
       
       pdata(c,4)=data.Input.Part(i).Geometry.Parameter; 
    end
end

% final check
if c==0
    flag=false;
    fprintf('Warning (Assembly): No active part detected!\n');
end

