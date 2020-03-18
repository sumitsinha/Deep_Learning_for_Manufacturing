function [idnode, flag]=getLocalActiveSelection(data, sdata, idparts)

% data: data structure
% sdata: list of selection (true/false) - [1, 2, ..., data.Input.Selection(ID),...]
    % They refer to data.Input.Selection
    % If "sdata" is empty => use all nodes in the model
% idparts: part list
% flag=0/1: no selection (use automatic selection)/use user selection

% init outputs
flag=1;

% read active selections
activesele=[];
for i=1:length(sdata)
  if sdata(i)
      activesele=[activesele, i]; %#ok<AGROW>
  end
end

%--
if isempty(activesele)
    flag=0;
    activesele=0;
end

% read parts
idnode=cell(1,length(idparts));
c=1;
for idparti=idparts
    idnodep=[];
    for i=1:length(activesele)
        if activesele(i)==0 % automatic
            idnodei=data.Model.Nominal.Domain(idparti).Node;
        else
            idnodei=getSelectionVolume(data, idparti, activesele(i));
        end

        idnodep=[idnodep, idnodei]; %#ok<AGROW>
    end
                
    idnode{c}=unique(idnodep);
    c=c+1;
end
