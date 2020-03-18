% map deviation field to RBG domain

% Use a linear interpolation between red (max deviation) and blu (min deviation). Green corresponds to the middle deviation.
function [R, G, B]=mapRGB(maxdev, mindev, devk)

% get red
R=mapRed(maxdev, mindev, devk);

% get green
G=mapGreen(maxdev, mindev, devk);

% get blu
B=mapBlu(maxdev, mindev, devk);

%--------------------------------
% map red
function R=mapRed(M, m, devk)

eps=1e-6;

mid=(m+M)/2;

if devk<=mid
    a=0.0;
    b=0.0;
else
    
    if abs(M-mid)<=eps
        a=0.0;
        b=0.0;
    else
        a=1/(M-mid);
        b=mid/(mid-M);
    end
    
end

R=a*devk+b;

% map green
function G=mapGreen(M, m, devk)

eps=1e-6;

mid=(m+M)/2;

if devk<=mid
    
    if abs(mid-m)<=eps
        a=0.0;
        b=0.0;
    else
        a=1/(mid-m);
        b=m/(m-mid);
    end
else
    if abs(mid-M)<=eps
        a=0.0;
        b=0.0;
    else
        a=1/(mid-M);
        b=M/(M-mid);
    end
end

G=a*devk+b;

% map blu
function B=mapBlu(M, m, devk)

eps=1e-6;

mid=(m+M)/2;

if devk<=mid
    if abs(m-mid)<=eps
        a=0.0;
        b=0.0;
    else
        a=1/(m-mid);
        b=mid/(mid-m);
    end
else
    a=0.0;
    b=0.0;
end

B=a*devk+b;




