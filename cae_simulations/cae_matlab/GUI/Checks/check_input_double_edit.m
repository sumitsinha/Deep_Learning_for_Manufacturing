% check is string is numeric value (only scalars are allowed)
function [numVal, flag]=check_input_double_edit(str, defVal, checkcond)

if nargin==2
    checkcond.Enable=false;
end

numVal=str2num(str); %#ok<ST2NM>

flag=true;
if ~isempty(numVal) 
    if isnan(numVal)
        numVal=defVal;
        flag=false;
    else
        if length(numVal)>1 % only scalars are allowed
           numVal=defVal;
           flag=false;
        else
           
           if checkcond.Enable % check constraint
               
               nc=length(checkcond.Type);
               for i=1:nc
                   if strcmp(checkcond.Type{i}, '<=')
                       if numVal>checkcond.b(i)
                           numVal=defVal;
                           flag=false;
                           return
                       end
                   elseif strcmp(checkcond.Type{i}, '>=')
                       if numVal<checkcond.b(i)
                           numVal=defVal;
                           flag=false;
                           return
                       end
                   elseif strcmp(checkcond.Type{i}, '==')
                       if numVal~=checkcond.b(i)
                           numVal=defVal;
                           flag=false;
                           return
                       end
                   elseif strcmp(checkcond.Type{i}, '<')
                       if numVal>=checkcond.b(i)
                           numVal=defVal;
                           flag=false;
                           return
                       end
                   elseif strcmp(checkcond.Type{i}, '>')
                       if numVal<=checkcond.b(i)
                           numVal=defVal;
                           flag=false;
                           return
                       end
                   end
               end
               
           end
        end
    end
else
  numVal=defVal;
  flag=false;  
end


