%--
function [flag, paraID]=checkInputStatusGivenPoint(f, paraID, idPOINT)

% Inputs:
% f: input field
% paraID: parameter ID (it may contain multiple values)
% idPOINT: point ID

% Outputs:
% flag: calculated/not calculated => true/false
% paraID: updated parameter ID
   
flag=false;

if length(f.Status)==1
    st=f.Status{1};
else
    st=f.Status{idPOINT};
end

if length(st)==1
    if st==0 && f.Enable % correctly calculated
        flag=true;
        paraID=1;
    end
    return
end

m=length(paraID); % no. of parameters
flag=false(1,m);

count=1;
for z=paraID
   if st(z)==0 && f.Enable % correctly calculated
       flag(count)=true;
   end
   count=count+1;
end 


