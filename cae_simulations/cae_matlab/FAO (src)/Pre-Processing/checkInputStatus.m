% Chek status of input field
function [flag, idPOINT]=checkInputStatus(f, paraID, field)

% Inputs:
% f: input field
% paraID: parameter ID (it may contain multiple values)
% field: field type

% Outputs:
% flag: calculated/not calculated => true/false
% idPOINT: ID of points calculated

if strcmp(field,'Selection') 
    flag=true;
    idPOINT=0; % not point is associated to "Selection"
    return
end

m=length(paraID); % no. of paramaters
flag=false(1,m);
idPOINT=zeros(1,m);
        
n=length(f.Status); % no. of points

tflag=false(n,m);
for j=1:n
   st=f.Status{j};
   count=1;
   for z=paraID
       if length(st)==1
          tst=st(1);                  
       else
          if z>length(st) % this check may needed when the no. of parameters within the same items varies
            tst=0;
          else
            tst=st(z);
          end
       end
       if tst==0 % correctly calculated
           tflag(j,count)=true;
       end
       count=count+1;  
   end 
   
end
%--
for z=1:m
   if strcmp(field,'Stitch')
        if all(tflag(:,z))
          flag(z)=true; 
        end
   else
        for j=1:n
           if tflag(j,z) % correctly calculated
                flag(z)=true;
                idPOINT(z)=j;
                break
           end
        end
   end
end

