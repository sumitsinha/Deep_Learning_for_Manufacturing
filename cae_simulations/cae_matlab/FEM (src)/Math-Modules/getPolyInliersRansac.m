% ransac noise reduction using polynomial fitting
function bestinliers=getPolyInliersRansac(X, Y, degree,...
                                                maxIter,...
                                                ssize,...
                                                t,...
                                                p)
                                  
% INPUT:
% X: independent variables [nsample, nvars]
% Y: dependent variables [nsample, 1] 
% degree: max polynomial degree
% maxIter: maximum ransac refiniments
% ssize: minimum number of samples
% t:  The distance threshold between a data point and the model used to decide whether the point is an inlier or not
% p: desired probability of choosing at least one sample free from outliers
     
% OUTPUT
% bestinliers: inliers ids

fprintf('Ransac de-noising: running...\n')

% no. of point the data set
nsample=size(X, 1);

bestinliers=1:nsample;
ninliers=nsample;

% run iterations
trialcount = 0;
bestscore =  0;
N = 1;          
while N > trialcount

    % STEP 1: random sample
    ind = randsample(nsample, ssize);

    % STEP 2: calculate model parameters that fit the data in the sample
    polymodel=getPolyFit(X(ind,:), Y(ind), degree);
       

    % STEP 3: calculate distance of the points from the model
    inliers=distmodel(polymodel, X, Y, t);
    
    % Find the number of inliers to this model.
    ninliers = length(inliers);

    % Largest set of inliers so far...
    if ninliers > bestscore    
        bestscore = ninliers;  % Record data for this model
        bestinliers = inliers;

        % Update estimate of N, the number of trials to ensure we pick, with probability p, a data set with no outliers.
        fracinliers =  ninliers/nsample;
        pNoOutliers = 1 -  fracinliers^ssize;
        pNoOutliers = max(eps, pNoOutliers);  % Avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers);% Avoid division by 0.
        N = log(1-p)/log(pNoOutliers);
    end
    
    %...
    if trialcount > maxIter
        warning('Ransac: ransac reached the maximum number!\n');
        break
    end
    
    trialcount = trialcount+1;
    
end
   
% plot outcomes
fprintf('------------------\n')
fprintf('Ransac de-noising: summary\n')
fprintf('      no. of RANSAC iterations %g\n', trialcount)
fprintf('      no. of inliers %g\n', bestscore)
fprintf('------------------\n')


%--------------------
function inliers=distmodel(polymodel, X, Y, t)

nsample=size(X,1);

inliers=1:nsample;

Ym=evalPolyFit(polymodel, X);

d=abs(Y-Ym);

% save out
inliers=inliers(d<=t); % might be empty




