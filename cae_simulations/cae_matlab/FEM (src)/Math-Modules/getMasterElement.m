function idele=getMasterElement(fem, mid)

idelet=fem.Sol.Node2Element{mid};

% loop over all elements
nele=length(idelet);
idt=[];
for i=1:nele
    idt=[idt,fem.xMesh.Element(idelet(i)).Element];    
end

ide=[];
for i=1:length(idt)
    ide=[ide,fem.Sol.Node2Element{idt(i)}];
end

idele=unique(ide);