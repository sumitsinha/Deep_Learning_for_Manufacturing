% plot input data structure
function plotDataInput(data, paraid, tag)

% data: data structure
% paraid: parameter ID
% tag: graphical ID for plotting purposes

if nargin<3
    tag='';
end

% Stitch
if isfield(data.database.Input, 'Stitch')
    n=length(data.database.Input.Stitch);
    for i=1:n
        plotDataInputSingle(data, 'Stitch', i, paraid, [], tag)
        plotDataInputSingleNormal(data, 'Stitch', i, paraid, tag)
    end
end

% Custom constraint
if isfield(data.database.Input, 'CustomConstraint')
    n=length(data.database.Input.CustomConstraint);
    for i=1:n
        plotDataInputSingle(data, 'CustomConstraint', i, paraid, [], tag)
        plotDataInputSingleNormal(data, 'CustomConstraint', i, paraid, tag)
    end
end

% Hole
if isfield(data.database.Input, 'PinLayout')
    if isfield(data.database.Input.PinLayout, 'Hole')
        n=length(data.database.Input.PinLayout.Hole);
        for i=1:n
            plotDataInputSingle(data, 'Hole', i, paraid, [], tag)
            plotDataInputSingleNormal(data, 'Hole', i, paraid, tag)
        end
    end
end

% Slot
if isfield(data.database.Input, 'PinLayout')
    if isfield(data.database.Input.PinLayout, 'Slot')
        n=length(data.database.Input.PinLayout.Slot);
        for i=1:n
            plotDataInputSingle(data, 'Slot', i,paraid, [], tag)
            plotDataInputSingleNormal(data, 'Slot', i, paraid, tag)
        end
    end
end

% NcBlock
if isfield(data.database.Input, 'Locator')
    if isfield(data.database.Input.Locator, 'NcBlock')
        n=length(data.database.Input.Locator.NcBlock);
        for i=1:n
            plotDataInputSingle(data, 'NcBlock', i, paraid, [], tag)
            plotDataInputSingleNormal(data, 'NcBlock', i, paraid, tag)
        end
    end
end

% ClampS
if isfield(data.database.Input, 'Locator')
    if isfield(data.database.Input.Locator, 'ClampS')
        n=length(data.database.Input.Locator.ClampS);
        for i=1:n
            plotDataInputSingle(data, 'ClampS', i, paraid, [], tag)
            plotDataInputSingleNormal(data, 'ClampS', i, paraid, tag)
        end
    end
end

% ClampM
if isfield(data.database.Input, 'Locator')
    if isfield(data.database.Input.Locator, 'ClampM')
        n=length(data.database.Input.Locator.ClampM);
        for i=1:n
            plotDataInputSingle(data, 'ClampM', i, paraid, [], tag)
            plotDataInputSingleNormal(data, 'ClampM', i, paraid, tag)
        end
    end
end
