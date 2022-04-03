function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
features = size(X,2);
 


for i =  1:length(X)
  X_k = zeros(K,features);
  %Fill K with x(i)'s
  X_k = X_k.+X(i,:);
  %Now X_k is a matrix with K number of elements (Same as no of centroids) where each element is X(i).

  mod_dist = (X_k - centroids) .^ 2;
  dist_matrix = sum(mod_dist,2);
  
  [minimum, min_index] = min(dist_matrix);

##  to debug
##  if (i == 3)
##    X_k
##    centroids
##    mod_dist
##    dist_matrix
##    minimum
##  endif

  idx(i) = min_index;
endfor






% =============================================================

end

