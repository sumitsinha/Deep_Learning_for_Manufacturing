% Set part UCS (only for locator placement)
function data=modelAssignUCSLocatorPlacement(data, stationData, stationID)

f=getInputFieldModel(data, 'Hole');
if isempty(f)
    return;
end
%
if stationData(stationID).Type{1}==2 % locator placement
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
                % Build UCS matrix
                Tucs=eye(4,4); Tucs(1:3,1:3)=R0h; Tucs(1:3,4)=P0h; 
                % save UCS back
                data.Input.Part(idpart).Placement.UCSStore{stationID}=Tucs; 
            end
        end
    end
end
