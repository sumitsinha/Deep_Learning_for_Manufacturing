% Add new item to pre-allocated model
function data=modelAddItem(data, flag)

%  
if strcmp(flag,'Parameter')
   if isfield(data.Input,'Parameter')
       c=length(data.Input.Parameter)+1;
   else
       c=1;
   end
   data.Input.Parameter(c)=initInputDatabase(flag);
elseif strcmp(flag,'Selection')
   if isfield(data.Input,'Selection')
       c=length(data.Input.Selection)+1;
   else
       c=1;
   end
   data.Input.Selection(c)=initInputDatabase(flag);
elseif strcmp(flag,'PartG')
   if isfield(data.Input,'PartG')
       c=length(data.Input.PartG)+1;
   else
       c=1;
   end
   data.Input.PartGraphic(c)=initInputDatabase(flag);
elseif strcmp(flag,'Part')
   if isfield(data.Input,'Part')
       c=length(data.Input.Part)+1;
   else
       c=1;
   end
   data.Input.Part(c)=initInputDatabase(flag); 
elseif strcmp(flag,'Robot')
   if isfield(data.Input,'Robot')
       c=length(data.Input.Robot)+1;
   else
       c=1;
   end
   data.Input.Robot(c)=initInputDatabase(flag);
elseif strcmp(flag,'Stitch')
   if isfield(data.Input,'Stitch')
       c=length(data.Input.Stitch)+1;
   else
       c=1;
   end
   data.Input.Stitch(c)=initInputDatabase(flag);    
elseif strcmp(flag,'Hole')
   if isfield(data.Input,'PinLayout')
       if isfield(data.Input.PinLayout,'Hole')
        c=length(data.Input.PinLayout.Hole)+1;
       else
        c=1;
       end
   else
       c=1;
   end
   data.Input.PinLayout.Hole(c)=initInputDatabase(flag);
elseif strcmp(flag,'Slot')
   if isfield(data.Input,'PinLayout')
       if isfield(data.Input.PinLayout,'Slot')
        c=length(data.Input.PinLayout.Slot)+1;
       else
        c=1;
       end
   else
       c=1;
   end
   data.Input.PinLayout.Slot(c)=initInputDatabase(flag);
elseif strcmp(flag,'NcBlock')
   if isfield(data.Input,'Locator')
       if isfield(data.Input.Locator,'NcBlock')
           c=length(data.Input.Locator.NcBlock)+1;
       else
           c=1;
       end
   else
     c=1;
   end
   data.Input.Locator.NcBlock(c)=initInputDatabase('Locator');
elseif strcmp(flag,'ClampS')
   if isfield(data.Input,'Locator')
       if isfield(data.Input.Locator,'ClampS')
           c=length(data.Input.Locator.ClampS)+1;
       else
           c=1;
       end
   else
     c=1;
   end
   data.Input.Locator.ClampS(c)=initInputDatabase('Locator'); 
elseif strcmp(flag,'ClampM')
   if isfield(data.Input,'Locator')
       if isfield(data.Input.Locator,'ClampM')
           c=length(data.Input.Locator.ClampM)+1;
       else
           c=1;
       end
   else
     c=1;
   end
   data.Input.Locator.ClampM(c)=initInputDatabase('Locator');
elseif strcmp(flag,'CustomConstraint')
   if isfield(data.Input,'CustomConstraint')
       c=length(data.Input.CustomConstraint)+1;
   else
       c=1;
   end
   data.Input.CustomConstraint(c)=initInputDatabase(flag); 
elseif strcmp(flag,'Contact')
   if isfield(data.Input,'Contact')
       c=length(data.Input.Contact)+1;
   else
       c=1;
   end
   data.Input.Contact(c)=initInputDatabase(flag);
elseif strcmp(flag,'Result')
   if isfield(data,'Result')
       c=length(data.Result)+1;
   else
       c=1;
   end
   data.Result(c)=initResultDatabase();
end


