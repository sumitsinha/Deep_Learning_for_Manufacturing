% Set active station
function data=modelStationSet(data, stationData, stationID)

% INPUTS:
% data: input model
% stationData(stationID): station data
% stationID: station ID (integer)
%
% OUTPUTS:
% data: input model with ".Enable" field udpated

%---
parID=1;
%---

%--------------------------------
% Enable Parts at current station
if isfield(data.Input,'Part')
    for i=1:length(data.Input.Part)
        flag=false;
        for j=1:length(stationData(stationID).Part)
             if i==stationData(stationID).Part(j)
                if data.Input.Part(i).Status==0
                    flag=true;
                    break
                end             
             end
        end
        data.Input.Part(i).Enable=flag;
    end
end
%
% Enable PinHoles at current station
if isfield(data.Input,'PinLayout')
    if isfield(data.Input.PinLayout, 'Hole')
        for i=1:length(data.Input.PinLayout.Hole)
            flag=false;
            for j=1:length(stationData(stationID).PinHole)
                 if i==stationData(stationID).PinHole(j)
                     % get field
                     [f, ~]=retrieveStructure(data, 'Hole', i);
                     [flagf, ~]=checkInputStatus(f, parID, 'Hole');
                     if flagf
                         [~, flagpass]=checkMasterStatus(data, f);
                         if flagpass
                             flag=true;
                             break
                         end
                     end
                 end
            end
            data.Input.PinLayout.Hole(i).Enable=flag; 
        end
    end
end
%
% Enable Slots at current station
if isfield(data.Input,'PinLayout')
    if isfield(data.Input.PinLayout, 'Slot')
        for i=1:length(data.Input.PinLayout.Slot)
            flag=false;
            for j=1:length(stationData(stationID).PinSlot)
                 if i==stationData(stationID).PinSlot(j)
                     % get field
                     [f, ~]=retrieveStructure(data, 'Slot', i);
                     [flagf, ~]=checkInputStatus(f, parID, 'Slot');
                     if flagf
                         [~, flagpass]=checkMasterStatus(data, f);
                         if flagpass
                             flag=true;
                             break
                         end
                     end
                 end
            end
            data.Input.PinLayout.Slot(i).Enable=flag;  
        end
    end
end
%
% Enable NcBlocks at current station
if isfield(data.Input, 'Locator')
    if isfield(data.Input.Locator, 'NcBlock')
        for i=1:length(data.Input.Locator.NcBlock)
            flag=false;
            for j=1:length(stationData(stationID).NcBlock)
                 if i==stationData(stationID).NcBlock(j)
                     % get field
                     [f, ~]=retrieveStructure(data, 'NcBlock', i);
                     [flagf, ~]=checkInputStatus(f, parID, 'NcBlock');
                     if flagf
                         [~, flagpass]=checkMasterStatus(data, f);
                         if flagpass
                             flag=true;
                             break
                         end
                     end
                 end
            end
            data.Input.Locator.NcBlock(i).Enable=flag;  
        end 
    end
end
%
% Enable ClampS at current station
if isfield(data.Input, 'Locator')
    if isfield(data.Input.Locator, 'ClampS')
        for i=1:length(data.Input.Locator.ClampS)
            flag=false;
            for j=1:length(stationData(stationID).ClampS)
                 if i==stationData(stationID).ClampS(j)
                     % get field
                     [f, ~]=retrieveStructure(data, 'ClampS', i);
                     [flagf, ~]=checkInputStatus(f, parID, 'ClampS');
                     if flagf
                         [~, flagpass]=checkMasterStatus(data, f);
                         if flagpass
                             flag=true;
                             break
                         end
                     end
                 end
            end
            data.Input.Locator.ClampS(i).Enable=flag;  
        end 
    end
end
%
% Enable ClampM at current station
if isfield(data.Input, 'Locator')
    if isfield(data.Input.Locator, 'ClampM')
        for i=1:length(data.Input.Locator.ClampM)
            flag=false;
            for j=1:length(stationData(stationID).ClampM)
                 if i==stationData(stationID).ClampM(j)
                     % get field
                     [f, ~]=retrieveStructure(data, 'ClampM', i);
                     [flagf, ~]=checkInputStatus(f, parID, 'ClampM');
                     if flagf
                         [~, flagpassM]=checkMasterStatus(data, f);
                         [~, flagpassS]=checkSlaveStatus(data, f);
                         if flagpassM && flagpassS
                             flag=true;
                             break
                         end
                     end
                 end
            end
            data.Input.Locator.ClampM(i).Enable=flag; 
        end 
    end
end
%
% Enable Custom constraints at current station
if isfield(data.Input, 'CustomConstraint')
    for i=1:length(data.Input.CustomConstraint)
        flag=false;
        for j=1:length(stationData(stationID).CustomConstraint)
             if i==stationData(stationID).CustomConstraint(j)
                 if i==stationData(stationID).CustomConstraint(j)
                     % get field
                     [f, ~]=retrieveStructure(data, 'CustomConstraint', i);
                     [flagf, ~]=checkInputStatus(f, parID, 'CustomConstraint');
                     if flagf
                         [~, flagpass]=checkMasterStatus(data, f);
                         if flagpass
                             flag=true;
                             break
                         end
                     end
                 end
             end
        end
        data.Input.CustomConstraint(i).Enable=flag;
    end
end
%
% Enable Stitch at current station
if isfield(data.Input, 'Stitch')
    for i=1:length(data.Input.Stitch)
        flag=false;
        for j=1:length(stationData(stationID).Stitch)
             if i==stationData(stationID).Stitch(j)
                 % get field
                 [f, ~]=retrieveStructure(data, 'Stitch', i);
                 [flagf, ~]=checkInputStatus(f, parID, 'Stitch');
                 if flagf
                     [~, flagpassM]=checkMasterStatus(data, f);
                     [~, flagpassS]=checkSlaveStatus(data, f);
                     if flagpassM && flagpassS
                         flag=true;
                         break
                     end
                 end
             end
        end
        data.Input.Stitch(i).Enable=flag;
    end
end
%
% Enable Contact Pairs at current station
if isfield(data.Input, 'Contact')
    for i=1:length(data.Input.Contact)
        flag=false;
        for j=1:length(stationData(stationID).Contact)
             if i==stationData(stationID).Contact(j)
                 % get field
                 [f, ~]=retrieveStructure(data, 'Contact', i);
                 [flagf, ~]=checkInputStatus(f, parID, 'Contact');
                 if flagf
                     [~, flagpassM]=checkMasterStatus(data, f);
                     [~, flagpassS]=checkSlaveStatus(data, f);
                     if flagpassM && flagpassS
                         flag=true;
                         break
                     end
                 end
             end
        end
        data.Input.Contact(i).Enable=flag;
    end
end

