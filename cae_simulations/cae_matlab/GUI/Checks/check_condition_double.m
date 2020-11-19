% check is numeric value satisfied conditions (only scalars are allowed)
function [numVal, flag]=check_condition_double(numVal, defVal, checkcond)

flag=true;

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
