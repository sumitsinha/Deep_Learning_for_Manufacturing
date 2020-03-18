function data=modelBuildInput(data, opt)

% opt: [logical]
    % opt(1): true/false => true: refresh all inputs (recompute projections, recompute parametrisations)
    % opt(2): true/false => true: refresh only part features (recompute projections, recompute parametrisations)
    % opt(3): true/false => true: apply placement
    
if nargin==1
    opt=true(1,3);
end

% Check part status
p=getInputFieldModel(data, 'Part');
for i=1:length(p)
    if p(i).Enable
        data.Input.Part(i).Status=0;
    else
        data.Input.Part(i).Status=-1;
    end
end

% Check input status
flag={'Stitch', 'Hole', 'Slot','NcBlock', 'ClampM','ClampS','CustomConstraint', 'Contact'};
nflags=length(flag);

for k=1:nflags
    f=getInputFieldModel(data, flag{k});
    % build
    n=length(f);
    for i=1:n
        data=updateDataInputSingle(data, flag{k}, i, opt);
    end
end

 