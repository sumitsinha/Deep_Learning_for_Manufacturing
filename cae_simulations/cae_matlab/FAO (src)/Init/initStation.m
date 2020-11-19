% Initialise station structure
function stationData=initStation()

% stationData(stationID): station data
    % .Part: [integer]
    % .PartFlipNormal: [boolean]
    
    % .UCS
        % .Source=[integer] => list of source parts to be linked with same UCS
        % .Destination=[integer] => list of destination parts to be linked with same UCS
        
    % .PinHole: [integer]
    % .PinSlot: [integer]
    % .NcBlock: [integer]
    % .ClampS: [integer]
    % .ClampM: [integer]
    % .CustomConstraint: [integer]
    % .Stitch: [integer]
    % .Contact: [integer]
    
    % .Type: station type
        % "Non Ideal Part" => non ideal part
        % "Rigid Placement" => placement using roto-traslation of parts
        % "Locator Placement" => placement using hole and slot
        % "Clamp" => clamping
        % "Release" => release
%
stationData.Part=[];
        % stationData.PartFlipNormal=[];

stationData.UCS.Source=[];
stationData.UCS.Destination=[];

stationData.PinHole=[];
stationData.PinSlot=[];
stationData.NcBlock=[];
stationData.ClampS=[];
stationData.ClampM=[];
stationData.CustomConstraint=[];
stationData.Stitch=[];
stationData.Contact=[];
%
stationData.Type={1,'Non Ideal Part',...
                    'Rigid Placement',...
                    'Locator Placement',...
                    'Clamp',...
                    'Release'};
