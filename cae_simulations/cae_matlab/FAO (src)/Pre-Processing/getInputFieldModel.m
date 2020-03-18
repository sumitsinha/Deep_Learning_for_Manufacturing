% get field
function f=getInputFieldModel(data, field)

f=[];
if strcmp(field,'Parameter') && isfield(data.Input,'Parameter')
    f=data.Input.Parameter;  
elseif strcmp(field,'Selection') && isfield(data.Input,'Selection')
    f=data.Input.Selection;  
elseif strcmp(field,'Part') && isfield(data.Input,'Part')
    f=data.Input.Part;  
elseif strcmp(field,'PartG') && isfield(data.Input,'PartGraphic')
    f=data.Input.PartGraphic; 
elseif strcmp(field,'Robot') && isfield(data.Input,'Robot')
    f=data.Input.Robot; 
elseif strcmp(field,'Stitch') && isfield(data.Input,'Stitch')
    f=data.Input.Stitch;    
elseif strcmp(field,'Hole')
    if isfield(data.Input,'PinLayout')
        if isfield(data.Input.PinLayout,'Hole')
            f=data.Input.PinLayout.Hole;
        end
    end
elseif strcmp(field,'Slot') 
    if isfield(data.Input,'PinLayout')
        if isfield(data.Input.PinLayout,'Slot')
            f=data.Input.PinLayout.Slot;
        end
    end
elseif strcmp(field,'CustomConstraint') && isfield(data.Input,'CustomConstraint')
    f=data.Input.CustomConstraint;     
elseif strcmp(field,'NcBlock') 
    if isfield(data.Input,'Locator')
        if isfield(data.Input.Locator,'NcBlock')
            f=data.Input.Locator.NcBlock;
        end
    end
elseif strcmp(field,'ClampS') 
    if isfield(data.Input,'Locator')
        if isfield(data.Input.Locator,'ClampS')
            f=data.Input.Locator.ClampS;
        end
    end
elseif strcmp(field,'ClampM') 
    if isfield(data.Input,'Locator')
        if isfield(data.Input.Locator,'ClampM')
            f=data.Input.Locator.ClampM;
        end
    end
elseif strcmp(field,'Contact') && isfield(data.Input,'Contact')
    f=data.Input.Contact;  
elseif strcmp(field,'Assembly') && isfield(data,'Assembly')
    f=data.Assembly; 
elseif strcmp(field,'Result') && isfield(data,'Result')
    f=data.Result; 
elseif strcmp(field,'Regression') && isfield(data,'Regression')
    f=data.Regression; 
elseif strcmp(field,'Optimisation') && isfield(data,'Optimisation')
    f=data.Optimisation; 
end
