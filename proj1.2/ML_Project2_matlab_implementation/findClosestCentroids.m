function memberships = findClosestCentroids(X, centroids)

% Set 'k' to the number of centers.
k = size(centroids, 1);

% Set 'm' to the number of data points.
m = size(X, 1);

% 'memberships' will hold the cluster numbers for each example.
memberships = zeros(m, 1);

% Create a matrix to hold the distances between each data point and
% each cluster center.
distances = zeros(m, k);

% For each cluster...
for i = 1 : k
    
    % Rather than compute the full euclidean distance, we just compute
    % the squared distance (i.e., ommit the sqrt) since this is sufficient
    % for performing distance comparisons.
    
    % Subtract centroid i from all data points.
    diffs = bsxfun(@minus, X, centroids(i, :));
    
    % Square the differences.
    sqrdDiffs = diffs .^ 2;
    
    % Take the sum of the squared differences.
    distances(:, i) = sum(sqrdDiffs, 2);

end

% Find the minimum distance value, also set the index of 
% the minimum distance value (in this case the indeces are 
% equal to the cluster numbers).
[minVals memberships] = min(distances, [], 2);

end

