% Reset active station
function data=modelStationReset(data)

% INPUTS:
% data: input model
% stationData(stationID): station data
% stationID: station ID (integer)
%
% OUTPUTS:
% data: reset input model with updated part features

%--------------------------------
% Reset solver settings
data.Assembly.Solver.UsePreLoad=false;
%--------------------------------

%--------------------------------
% Reset Parts at current station
if isfield(data.Input,'Part')
    for i=1:length(data.Input.Part)
        data.Input.Part(i).Enable=data.Input.Part(i).EnableReset;
        data.Input.Part(i).Status=0;
    end
end
%
% Reset PinHoles at current station
if isfield(data.Input,'PinLayout')
    if isfield(data.Input.PinLayout, 'Hole')
        for i=1:length(data.Input.PinLayout.Hole)
            data.Input.PinLayout.Hole(i).Enable=data.Input.PinLayout.Hole(i).EnableReset;  
        end
    end
end
%
% Reset Slots at current station
if isfield(data.Input,'PinLayout')
    if isfield(data.Input.PinLayout, 'Slot')
        for i=1:length(data.Input.PinLayout.Slot)
            data.Input.PinLayout.Slot(i).Enable=data.Input.PinLayout.Slot(i).EnableReset;  
        end
    end
end
%
% Reset NcBlocks at current station
if isfield(data.Input, 'Locator')
    if isfield(data.Input.Locator, 'NcBlock')
        for i=1:length(data.Input.Locator.NcBlock)
            data.Input.Locator.NcBlock(i).Enable=data.Input.Locator.NcBlock(i).EnableReset;  
        end 
    end
end
%
% Reset ClampS at current station
if isfield(data.Input, 'Locator')
    if isfield(data.Input.Locator, 'ClampS')
        for i=1:length(data.Input.Locator.ClampS)
            data.Input.Locator.ClampS(i).Enable=data.Input.Locator.ClampS(i).EnableReset; 
        end 
    end
end
%
% Reset ClampM at current station
if isfield(data.Input, 'Locator')
    if isfield(data.Input.Locator, 'ClampM')
        for i=1:length(data.Input.Locator.ClampM)
            data.Input.Locator.ClampM(i).Enable=data.Input.Locator.ClampM(i).EnableReset; 
        end 
    end
end
%
% Reset Custom constraints at current station
if isfield(data.Input, 'CustomConstraint')
    for i=1:length(data.Input.CustomConstraint)
        data.Input.CustomConstraint(i).Enable=data.Input.CustomConstraint(i).EnableReset;
    end
end
%
% Reset Stitch at current station
if isfield(data.Input, 'Stitch')
    for i=1:length(data.Input.Stitch)
        data.Input.Stitch(i).Enable=data.Input.Stitch(i).EnableReset;
    end
end
%
% Reset Contact Pairs at current station
if isfield(data.Input, 'Contact')
    for i=1:length(data.Input.Contact)
        data.Input.Contact(i).Enable=data.Input.Contact(i).EnableReset;
    end
end
%
% Re-build
data=modelBuildInput(data,[0 1 0]);
%----------------


