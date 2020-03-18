% SWAP selected Component
   
%..
function fem=femSwapComponents(fem,...
                                 idcomp1, idcomp2)


%-
temp=fem.Domain(idcomp1);
fem.Domain(idcomp1)=fem.Domain(idcomp2);
fem.Domain(idcomp2)=temp;