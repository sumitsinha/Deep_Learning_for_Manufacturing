% Set part UCS (aligned to the hole)
function data=modelAssignUCSLocatorPlacement(data, stationData, stationID)

f=getInputFieldModel(data, 'Hole');
if isempty(f)
    return;
end
%
idHole=stationData(stationID).PinHole;
for i=1:length(idHole)
    idh=idHole(i);
    if f(idh).Status{1}==0 && f(idh).Enable % active hole
        % part ID
        idpart=f(idh).Master;
        if data.Input.Part(idpart).Status==0 && data.Input.Part(idpart).Enable
            % Rotation matrix
            R0h=f(idh).Parametrisation.Geometry.R{1};
            % Position vector
            P0h=f(idh).Pm;
            % Buid UCS matrix
            Tucs=eye(4,4); Tucs(1:3,1:3)=R0h; Tucs(1:3,4)=P0h; 
            % save UCS back
            data.Input.Part(idpart).Placement.UCS=Tucs; 
        end
    end
end
