function [xy, xyinit] = graphLayout(A, eps, stopC)

%- A: incidence matrix
%- eps: error allowed
%- stopC: max. n. of iterations allowed
%- xy: optimum vertex position

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%- This routine is partially based on:
%- 1. "Drawing Graph: Methods and Models, 2001"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%- Copyright, Pasquale Franciosa, December, 1-21 2009 - MIT.

fprintf('Graph Layout: calculating optimum 2D layout...\n');

%-use a damping factor
t=1;

nV=size(A,1);

%- get intial random value
xyinit=rand(nV,2);
xy=xyinit;

%- get stiffness constant
k = sqrt(1/nV); %- ... as suggested by the authors

flag=true;
count=0;
while  flag

    %- get initial value
    disp = zeros(nV,2);
    
    checkeps=0; %-error check
    for i = 1:nV
        for j=1:nV
          d = xy(i,:)-xy(j,:);
            if A(i,j)==1
                disp(i,:) = disp(i,:)-d/norm(d)*evalfa(norm(d),k); %-attractive force
                disp(j,:) = disp(j,:)+d/norm(d)*evalfa(norm(d),k);
            elseif A(i,j)==0 && i~=j
                disp(i,:) = disp(i,:)+d/norm(d)*evalfr(norm(d),k); %-repulsive force
            end
        end
    end
           
    for i = 1:nV  
        d=disp(i,:);
        td(1)=disp(i,1)/norm(disp(i,:));
        td(2)=disp(i,2)/norm(disp(i,:));
        
        xy(i,:) = xy(i,:) + (d/norm(d))*min(norm(d),t); %-update vertex position
                
        checkeps=checkeps+(norm(td)*min(norm(d),t))^2; %- update error checker
         
    end
    
    t=.99*t; %-reduce damping factor
      
    if sqrt(checkeps)<=eps %-stop when the error is less than eps
        fprintf('   solution converging after: %g iterations\n', count);
        break
    end
    
    if count>=stopC %-stop when the # of iterations if greater than stopC
        fprintf('Graph Layout (error): solution not converging!\n');
        break
    end

    count=count+1;

end

%- attractive force
function fa = evalfa(d,k)

fa = d^2/k;

%- repulsive force
function fr = evalfr(d,k)

fr = k^2/d;


