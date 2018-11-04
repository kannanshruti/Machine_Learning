%% Creating D3
D1 = sample_circle(3, [500, 500, 500]);
D2 = sample_spiral(3, [500, 500, 500]);
n = size(D1,1);

[theta,rho] = cart2pol(D1(:,1),D1(:,2));
theta = theta - min(theta);
theta = theta/ max(theta);
rho = rho - min(rho);
rho = rho/ max(rho);
D3 = [theta,rho]

%% Clustering
rng(2);
[idx, C] = kmeans(D3, 2, 'Replicates', 20, 'Distance', 'cityblock'); % k = 2
figure;
subplot(3,1,1);
plot(D3(idx==1,1), D3(idx==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D3(idx==2,1), D3(idx==2,2), 'b.', 'MarkerSize', 10); 
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 20, 'LineWidth', 5);
legend('Cluster1', 'Cluster2', 'Centroids');
title('D3 k = 2');
xlabel('Angle - Theta');
ylabel('Radius - Rho');
dist(1) = sum(sqrt((D3(idx==1,1)-C(1,1)).^2 + (D3(idx==1,2)-C(1,2)).^2));
dist(2) = sum(sqrt((D3(idx==2,1)-C(2,1)).^2 + (D3(idx==2,2)-C(2,2)).^2));

rng(2);
[idx, C] = kmeans(D3, 3, 'Replicates', 20, 'Distance', 'cityblock'); % k = 3
subplot(3,1,2);
plot(D3(idx==1,1), D3(idx==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D3(idx==2,1), D3(idx==2,2), 'g.', 'MarkerSize', 10); hold on;
plot(D3(idx==3,1), D3(idx==3,2), 'b.', 'MarkerSize', 10); 
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 20, 'LineWidth', 5);
legend('Cluster1', 'Cluster2', 'Cluster3', 'Centroids');
title('D3 k = 3');
xlabel('Angle - Theta');
ylabel('Radius - Rho');
dist(3) = sum(sqrt((D3(idx==1,1)-C(1,1)).^2 + (D3(idx==1,2)-C(1,2)).^2));
dist(4) = sum(sqrt((D3(idx==2,1)-C(2,1)).^2 + (D3(idx==2,2)-C(2,2)).^2));
dist(5) = sum(sqrt((D3(idx==3,1)-C(3,1)).^2 + (D3(idx==3,2)-C(3,2)).^2));

rng(2);
[idx, C] = kmeans(D3, 4, 'Replicates', 20, 'Distance', 'cityblock'); % k = 4
subplot(3,1,3);
plot(D3(idx==1,1), D3(idx==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D3(idx==2,1), D3(idx==2,2), 'g.', 'MarkerSize', 10); hold on;
plot(D3(idx==3,1), D3(idx==3,2), 'b.', 'MarkerSize', 10); hold on;
plot(D3(idx==4,1), D3(idx==4,2), 'k.', 'MarkerSize', 10); 
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 20, 'LineWidth', 5);
legend('Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Centroids');
title('D3 k = 4');
xlabel('Angle - Theta');
ylabel('Radius - Rho');
dist(6) = sum(sqrt((D3(idx==1,1)-C(1,1)).^2 + (D3(idx==1,2)-C(1,2)).^2));
dist(7) = sum(sqrt((D3(idx==2,1)-C(2,1)).^2 + (D3(idx==2,2)-C(2,2)).^2));
dist(8) = sum(sqrt((D3(idx==3,1)-C(3,1)).^2 + (D3(idx==3,2)-C(3,2)).^2));
dist(9) = sum(sqrt((D3(idx==4,1)-C(4,1)).^2 + (D3(idx==4,2)-C(4,2)).^2));