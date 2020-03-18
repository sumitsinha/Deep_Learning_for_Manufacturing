% Check Master & Slave
function data=checkMasterSlave(data, f, field, id, optCheck)

% data:data structure
% f: input field
% field: type of field
% id: input ID
% optCheck
    % 1 => check master
	% 2 => check master & slave (default option)

%---
if nargin==4
    optCheck=2;
end

if f.Enable
    % check master
    [f, flagpass]=checkMasterStatus(data, f);
    if ~flagpass
        data=retrieveBackStructure(data, f, field, id);
        return
    end
    if optCheck==2
        % check slave
        [f, flagpass]=checkSlaveStatus(data, f);
        if ~flagpass
            data=retrieveBackStructure(data, f, field, id);
            return
        end
    end
else
     for i=1:length(f.Status)
        f.Status{i}=-1;
     end
end
data=retrieveBackStructure(data, f, field, id);
