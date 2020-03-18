%--
function data=retrieveBackStructure(data, f, field, id)

if strcmp(field,'Parameter') && isfield(data.Input,'Parameter')
    data.Input.Parameter(id)=f;  
elseif strcmp(field,'Selection') && isfield(data.Input,'Selection')
    data.Input.Selection(id)=f; 
elseif strcmp(field,'PartG') && isfield(data.Input,'PartGraphic')
    data.Input.PartGraphic(id)=f; 
elseif strcmp(field,'Part') && isfield(data.Input,'Part')
    data.Input.Part(id)=f;  
elseif strcmp(field,'Robot') && isfield(data.Input,'Robot')
    data.Input.Robot(id)=f;  
elseif strcmp(field,'Stitch') && isfield(data.Input,'Stitch')
    data.Input.Stitch(id)=f;   
elseif strcmp(field,'Dimple') && isfield(data.Input,'Dimple')
    data.Input.Dimple(id)=f;  
elseif strcmp(field,'Hole')
    if isfield(data.Input,'PinLayout')
        if isfield(data.Input.PinLayout,'Hole')
            data.Input.PinLayout.Hole(id)=f;
        end
    end
elseif strcmp(field,'Slot') 
    if isfield(data.Input,'PinLayout')
        if isfield(data.Input.PinLayout,'Slot')
            data.Input.PinLayout.Slot(id)=f;
        end
    end
elseif strcmp(field,'CustomConstraint') && isfield(data.Input,'CustomConstraint')
    data.Input.CustomConstraint(id)=f;     
elseif strcmp(field,'NcBlock') 
    if isfield(data.Input,'Locator')
        if isfield(data.Input.Locator,'NcBlock')
            data.Input.Locator.NcBlock(id)=f;
        end
    end
elseif strcmp(field,'ClampS') 
    if isfield(data.Input,'Locator')
        if isfield(data.Input.Locator,'ClampS')
            data.Input.Locator.ClampS(id)=f;
        end
    end
elseif strcmp(field,'ClampM') 
    if isfield(data.Input,'Locator')
        if isfield(data.Input.Locator,'ClampM')
            data.Input.Locator.ClampM(id)=f;
        end
    end
elseif strcmp(field,'Contact') && isfield(data.Input,'Contact')
    data.Input.Contact(id)=f;  
elseif strcmp(field,'Assembly') && isfield(data,'Assembly')
    data.Assembly=f; 
elseif strcmp(field,'Result') && isfield(data,'Result')
    data.Result(id)=f; 
elseif strcmp(field,'Regression') && isfield(data,'Regression')
    data.Regression(id)=f; 
elseif strcmp(field,'Optimisation') && isfield(data,'Optimisation')
    data.Optimisation(id)=f; 
end
