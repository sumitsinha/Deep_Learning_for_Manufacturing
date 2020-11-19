% check is string is numeric value
function [numVal, flag]=check_input_double_vector_edit(str, defVal)

numVal=str2num(str); %#ok<ST2NM>

flag=true;
if ~isempty(numVal) 
    if isnan(numVal)
        numVal=defVal;
        flag=false;
    end
else
  numVal=defVal;
  flag=false;  
end
