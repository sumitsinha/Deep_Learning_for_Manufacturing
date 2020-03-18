%--
function [f, flag]=retrieveStructure(data, field, id)

flag=false;
f=[];

if strcmp(field,'Parameter') && isfield(data.Input,'Parameter')
    f=data.Input.Parameter(id); 
elseif strcmp(field,'Selection') && isfield(data.Input,'Selection')
    f=data.Input.Selection(id); 
elseif strcmp(field,'PartG') && isfield(data.Input,'PartGraphic')
    f=data.Input.PartGraphic(id);  
elseif strcmp(field,'Part') && isfield(data.Input,'Part')
    f=data.Input.Part(id);  
elseif strcmp(field,'Robot') && isfield(data.Input,'Robot')
    f=data.Input.Robot(id);
elseif strcmp(field,'Stitch') && isfield(data.Input,'Stitch')
    f=data.Input.Stitch(id);   
elseif strcmp(field,'Dimple') && isfield(data.Input,'Dimple')
    f=data.Input.Dimple(id);  
elseif strcmp(field,'Hole')
    if isfield(data.Input,'PinLayout')
        if isfield(data.Input.PinLayout,'Hole')
            f=data.Input.PinLayout.Hole(id);
        end
    end
elseif strcmp(field,'Slot') 
    if isfield(data.Input,'PinLayout')
        if isfield(data.Input.PinLayout,'Slot')
            f=data.Input.PinLayout.Slot(id);
        end
    end
elseif strcmp(field,'CustomConstraint') && isfield(data.Input,'CustomConstraint')
    f=data.Input.CustomConstraint(id);     
elseif strcmp(field,'NcBlock') 
    if isfield(data.Input,'Locator')
        if isfield(data.Input.Locator,'NcBlock')
            f=data.Input.Locator.NcBlock(id);
        end
    end
elseif strcmp(field,'ClampS') 
    if isfield(data.Input,'Locator')
        if isfield(data.Input.Locator,'ClampS')
            f=data.Input.Locator.ClampS(id);
        end
    end
elseif strcmp(field,'ClampM') 
    if isfield(data.Input,'Locator')
        if isfield(data.Input.Locator,'ClampM')
            f=data.Input.Locator.ClampM(id);
        end
    end
elseif strcmp(field,'Contact') && isfield(data.Input,'Contact')
    f=data.Input.Contact(id);  
elseif strcmp(field,'Result') && isfield(data,'Result')
    f=data.Result(id);     
elseif strcmp(field,'Assembly') && isfield(data,'Assembly')
    f=data.Assembly;  
elseif strcmp(field,'Regression') && isfield(data,'Regression')
    f=data.Regression(id);  
elseif strcmp(field,'Optimisation') && isfield(data,'Optimisation')
    f=data.Optimisation(id); 
end

if strcmp(field,'PartG') || strcmp(field,'Selection')
    flag=true;
    return
end

if ~isempty(f) && ~strcmp(field,'Parameter') && ~strcmp(field,'Selection') && ~strcmp(field,'PartG') && ~strcmp(field,'Assembly') && ~strcmp(field,'Result') && ~strcmp(field,'Regression') && ~strcmp(field,'Optimisation') 
    if f.Enable
        flag=true;
    end
end


