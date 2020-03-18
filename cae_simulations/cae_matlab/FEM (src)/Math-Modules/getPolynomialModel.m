function polymodel=getPolynomialModel(X, Y, degree,...
                                      numFolds,...
                                      numToLeaveOut)

% INPUT:
% X: independent variables [nsample, nvars]
% Y: dependent variables [nsample, 1] 
% degree: [min max] polynomial degree
% numFolds: no. of cross validations
% numToLeaveOut: no. of point to leave out

% OUTPUT
% polymodel
%        polymodel.ModelTerms = list of terms in the model
%        polymodel.Degree = polynomial degree
%        polymodel.Coefficients = regression coefficients
%        polymodel.R2 = coefficient of determination (R^2)
%        polymodel.RMSE = Root mean squared error

polymodel=[];

% check inputs
if degree(1)<0 || degree(2)<0
    fprintf('Polynomial fitting: degree must be bigger than zero!\n')
    return
end

if isempty(X) || isempty(Y)
    fprintf('Polynomial fitting: data set must be not empty!\n')
    return
end

if numToLeaveOut>=size(X,1)
    fprintf('Polynomial fitting: no. of leave-out too large!\n')
    return
end

if size(X,1)~=size(Y,1)
    fprintf('Polynomial fitting: data set not consistent!\n')
    return
end

if numFolds<0
    fprintf('Polynomial fitting: no. of cross validations must be bigger than 0!\n')
    return
end


% run...
fprintf('Polynomial fitting: running...\n')

% get size of the data set
[nsample, nvars]=size(X);
    
orders=degree(1):degree(2);
count=length(orders);

R2=zeros(1,count);
polymodel=cell(1,count);
c=1;
for iord=orders
    
    fprintf('   polynomial degree: %g\n',iord)
    
    % STEP 1: build the prediction model
    polymodel{c} = getPolyFit(X, Y, iord);
    
    % check
    if isnan(norm(polymodel{c}.Coefficients))
    
        polymodel{c}.R2 = 0;
        R2(c)=polymodel{c}.R2;
        
        polymodel{c}.RMSE=inf;

    else
        % ... now run cross validation to check the accuracy
        Ytest=zeros(1,numFolds*numToLeaveOut);
        Ym=zeros(1,numFolds*numToLeaveOut);
        ce=0;
        for k = 1:numFolds

             % STEP 2: split data into training and testing data
             [train, test] = crossvalind('LeaveMOut',nsample, numToLeaveOut);

             % STEP 3: calculate model
             Xtrain=X(train,:);
             Ytrain=Y(train,:);
             polymodelk = getPolyFit(Xtrain, Ytrain, iord);

             % STEP 4: calculate residuals
             Xtest=X(test,:);
             tYtest=Y(test,:);

             cs=ce+1;       
             ce=ce+length(tYtest);

             Ytest(cs:ce)=tYtest;

             Ym(cs:ce)=evalPolyFit(polymodelk, Xtest);

        end

        %---
        if numFolds==0
            R2(c)=polymodel{c}.R2;
        else
            
            % evaluate R^2
            SSr = norm(Ytest - Ym)^2;
                            % SSt=norm(Ytest-mean(Ytest))^2; % centered
            SSt=norm(Ytest)^2; % not centered

            polymodel{c}.R2 = max(0,1 - SSr/SSt);
            R2(c)=polymodel{c}.R2;

            % get root mean square error
            polymodel{c}.RMSE = sqrt(mean((Ytest - Ym).^2));
            
        end
    
    end
    
    % plot outcomes
    fprintf('      R2: %f\n',polymodel{c}.R2)
    fprintf('      RMS - Root Mean Square: %f\n',polymodel{c}.RMSE)
    
    c=c+1;

end

% get final model
[~, iord]=max(R2);
polymodel = polymodel{iord};

% plot outcomes
fprintf('------------------\n')
fprintf('Polynomial fitting: summary\n')
fprintf('      polynomial degree: %g\n',polymodel.Degree)
fprintf('      no. of cross validation: %g\n',numFolds)
fprintf('      no. of leave-out points: %g\n',numToLeaveOut)
fprintf('      no. of independent variables: %g\n',nvars)
fprintf('      no. data points: %g\n',nsample)
fprintf('      R2: %f\n',polymodel.R2)
fprintf('      RMS - Root Mean Square: %f\n',polymodel.RMSE)
fprintf('------------------\n')


 