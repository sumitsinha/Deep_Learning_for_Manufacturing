function plotFrame(R, P, ax, lsymbol, tag)

if nargin==4
    tag='tempobj';
end
    
% X
renderAxis(R(:,1)', P, ax, lsymbol, tag, 'r');

% Y
renderAxis(R(:,2)', P, ax, lsymbol, tag, 'g');

% Z
renderAxis(R(:,3)', P, ax, lsymbol, tag, 'b');
