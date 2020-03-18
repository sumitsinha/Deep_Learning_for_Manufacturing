function varargout=planeLineIntersection(Pp, Np, Pline, Nline)

% Pp/Np: plane parameters
% Pline, Nline: line parameters

nlnp=dot(Np, Nline);

if nlnp~=0
    t=dot( (Pp-Pline), Np )/nlnp;
    Pint=Pline+t*Nline;
    flag=true;
else
    Pint=[0 0 0];
    flag=false;
end

if nargout==1
    varargout{1}=Pint;
elseif nargout==2
    varargout{1}=Pint;
    varargout{2}=flag;
end

