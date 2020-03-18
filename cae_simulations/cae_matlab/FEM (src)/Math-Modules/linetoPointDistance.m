function d=linetoPointDistance(Pline, Nline, Pt)

% Pline, Nline: line parameters
% Pt: testing point

t=dot( (Pt-Pline), Nline );
Pi=Pline+t*Nline;

d=norm(Pt-Pi);