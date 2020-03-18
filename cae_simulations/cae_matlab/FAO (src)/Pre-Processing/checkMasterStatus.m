% Check master status
function [f, flagpass]=checkMasterStatus(data, f)

npart=length(data.Input.Part);

%----
ns=length(f.Status);
flagpass=true;
%----

if f.Master<=0 || f.Master>npart
    for i=1:ns
        f.Status{i}=1;
    end
    flagpass=false;
    return
end  

if isfield(f, 'DomainM')
    if f.DomainM<=0 || f.DomainM>npart
        for i=1:ns
            f.Status{i}=1;
        end
        flagpass=false;
        return
    end  
end

if data.Input.Part(f.Master).Status~=0 || ~data.Input.Part(f.Master).Enable
    for i=1:ns
        f.Status{i}=1;
    end
    flagpass=false;
    return
end

if isfield(f, 'DomainM')
    if data.Input.Part(f.DomainM).Status~=0 || ~data.Input.Part(f.DomainM).Enable
        for i=1:ns
            f.Status{i}=1;
        end
        flagpass=false;
        return
    end
end
