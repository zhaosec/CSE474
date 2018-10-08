function centroids = computeCentroids(X, prev_centroids, memberships, k)

[m n] = size(X);

centroids = zeros(k, n);

% For each centroid...
for (i = 1 : k)
    % If no points are assigned to the centroid, don't move it.
    if (~any(memberships == i))
        centroids(i, :) = prev_centroids(i, :);
    % Otherwise, compute the cluster's new centroid.
    else
        % Select the data points assigned to centroid k.
        points = X((memberships == i), :);

        % Compute the new centroid as the mean of the data points.
        centroids(i, :) = mean(points);    
    end
end

end

