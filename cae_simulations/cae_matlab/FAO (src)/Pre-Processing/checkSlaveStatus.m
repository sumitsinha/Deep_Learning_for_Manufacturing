% Check slave status
function [f, flagpass]=checkSlaveStatus(data, f)

npart=length(data.Input.Part);

%----
ns=length(f.Status);
flagpass=true;
%----

if f.Slave<=0 || f.Slave>npart || f.Slave==f.Master
    for i=1:ns
        f.Status{i}=2;
    end
    flagpass=false;
    return
end  

if isfield(f, 'DomainS')
    if f.DomainS<=0 || f.DomainS>npart || f.DomainS==f.DomainM
        for i=1:ns
            f.Status{i}=2;
        end
        flagpass=false;
        return
    end  
end

if data.Input.Part(f.Slave).Status~=0 || ~data.Input.Part(f.Slave).Enable
    for i=1:ns
        f.Status{i}=2;
    end
    flagpass=false;
    return
end

if isfield(f, 'DomainS')
    if data.Input.Part(f.DomainS).Status~=0 || ~data.Input.Part(f.DomainS).Enable
        for i=1:ns
            f.Status{i}=2;
        end
        flagpass=false;
        return
    end
end

