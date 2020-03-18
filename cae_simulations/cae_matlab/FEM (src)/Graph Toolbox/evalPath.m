%- find path connecting vertex Vi to Vj one
function P=evalPath(E, Vi, Vj)

%- E: edge matrix
%- Vi: start vertex
%- Vj: end vertex
%- P: path matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%- This routine is partially based on:
%- 2. "Graph and Hypergraphs, Berge, 1974"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%- Copyright, Pasquale Franciosa, December, 1-12 2009 - MIT.

nE=size(E,1);
nV=max(E(:));

I=edge2Incidence(E);

%- getting initial value
tagVertex=zeros(1,nV);
tagEdge=zeros(1,nE);

listVtx=Vi;
listEdge=[];
P=zeros(1,nE); 

while listVtx(end)~=Vj %- repeat until reaching the path condition (Vi=Vj)

  for k=1:nE
      
    fk=listVtx(end); %- starting vertex
      
    if E(k,1)==fk 
        hk=E(k,2); %- corresponding ending vertex
    else
        hk=E(k,1); %- corresponding ending vertex
    end
    
    if abs(I(fk,k))==1 %- connected edge
        if tagEdge(k)==0 %- never visited
            if tagVertex(hk)==0 %- never visited
                
                %- update path
                listVtx=[listVtx,hk];
                listEdge=[listEdge,k];
                
                P(k)=I(fk,k);
                
                tagVertex(fk)=tagVertex(fk)+1;
                tagEdge(k)=tagEdge(k)+1;
                
                if hk==Vj %- return function when the path condition is reached
                    return
                end
                
            end
        else %- move back
            
              %if ~isempty(listEdge) %-avoiding run-time error
                P(listEdge(end))=0;
                
                listVtx(end)=[];
                listEdge(end)=[];
                
                tagVertex(fk)=tagVertex(fk)+1;
                tagEdge(k)=tagEdge(k)+1;
              %end
            
        end       
    end
  end

end

    
