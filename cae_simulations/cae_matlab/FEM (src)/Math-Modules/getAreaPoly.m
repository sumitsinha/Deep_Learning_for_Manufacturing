% get area
function area=getAreaPoly(P)

np=size(P,1);

if np==3 % triangle
   area=getAreaTria(P);
else
    
   area=0;
   Pi=[P(1,:)
       P(2,:)
       P(3,:)];
   area=area+getAreaTria(Pi);
   
   Pi=[P(1,:)
       P(3,:)
       P(4,:)];
   area=area+getAreaTria(Pi);
   
end

