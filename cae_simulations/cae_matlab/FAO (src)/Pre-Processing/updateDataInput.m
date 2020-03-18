% update input structure
function data=updateDataInput(data)

% INPUT
% data: data structure

% Stitch
n=length(data.Input.Stitch);
for i=1:n
    data=updateDataInputSingle(data, 'Stitch', i);
end

% Custom constraint
n=length(data.Input.CustomConstraint);
for i=1:n
    data=updateDataInputSingle(data, 'CustomConstraint', i);
end

% Hole
n=length(data.Input.PinLayout.Hole);
for i=1:n
    data=updateDataInputSingle(data, 'Hole', i);
end

% Slot
n=length(data.Input.PinLayout.Slot);
for i=1:n
    data=updateDataInputSingle(data, 'Slot', i);
end

% NcBlock
n=length(data.Input.Locator.NcBlock);
for i=1:n
    data=updateDataInputSingle(data, 'NcBlock', i);
end

% ClampS
n=length(data.Input.Locator.ClampS);
for i=1:n
    data=updateDataInputSingle(data, 'ClampS', i);
end

% ClampM
n=length(data.Input.Locator.ClampM);
for i=1:n
    data=updateDataInputSingle(data, 'ClampM', i);
end
