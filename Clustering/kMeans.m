% Description: K-means clustering
%% Creating Clusters
D1 = sample_circle(3, [500, 500, 500]);
D2 = sample_spiral(3, [500, 500, 500]);
rng(2);
k = [2,3,4];

%% Clustering the data- D1 and D2
rng(2);
[idx1, C1] = kmeans(D1, k(1), 'Replicates', 20, 'Distance', 'sqeuclidean'); % D1, k=2
figure;
subplot(3,2,1);
plot(D1(idx1==1,1), D1(idx1==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D1(idx1==2,1), D1(idx1==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(C1(:,1), C1(:,2), 'kx', 'MarkerSize', 20, 'LineWidth', 5);
legend('Cluster1', 'Cluster2', 'Centroids');
title('Dataset D1: k = 2');
xlabel('Feature 1');
ylabel('Feature 2');
dist1(1) = sum(sqrt((D1(idx1==1,1)-C1(1,1)).^2 + (D1(idx1==1,2)-C1(1,2)).^2));
dist1(2) = sum(sqrt((D1(idx1==2,1)-C1(2,1)).^2 + (D1(idx1==2,2)-C1(2,2)).^2));

rng(2);
[idx1, C1] = kmeans(D1, k(2), 'Replicates', 20, 'Distance', 'sqeuclidean'); % D1, k=3
subplot(3,2,3);
plot(D1(idx1==1,1), D1(idx1==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D1(idx1==2,1), D1(idx1==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(D1(idx1==3,1), D1(idx1==3,2), 'g.', 'MarkerSize', 10); hold on;
plot(C1(:,1), C1(:,2), 'kx', 'MarkerSize', 20, 'LineWidth', 5);
legend('Cluster1', 'Cluster2', 'Cluster3', 'Centroids');
title('Dataset D1: k = 3');
xlabel('Feature 1');
ylabel('Feature 2');
dist1(3) = sum(sqrt((D1(idx1==1,1)-C1(1,1)).^2 + (D1(idx1==1,2)-C1(1,2)).^2));
dist1(4) = sum(sqrt((D1(idx1==2,1)-C1(2,1)).^2 + (D1(idx1==2,2)-C1(2,2)).^2));
dist1(5) = sum(sqrt((D1(idx1==3,1)-C1(3,1)).^2 + (D1(idx1==3,2)-C1(3,2)).^2));

rng(2);
[idx1, C1] = kmeans(D1, k(3), 'Replicates', 20, 'Distance', 'sqeuclidean'); % D1, k=4
subplot(3,2,5);
plot(D1(idx1==1,1), D1(idx1==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D1(idx1==2,1), D1(idx1==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(D1(idx1==3,1), D1(idx1==3,2), 'g.', 'MarkerSize', 10); hold on;
plot(D1(idx1==4,1), D1(idx1==4,2), 'k.', 'MarkerSize', 10); hold on;
plot(C1(:,1), C1(:,2), 'kx', 'MarkerSize', 20, 'LineWidth', 5);
legend('Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Centroids');
title('Dataset D1: k = 4');
xlabel('Feature 1');
ylabel('Feature 2');     
dist1(6) = sum(sqrt((D1(idx1==1,1)-C1(1,1)).^2 + (D1(idx1==1,2)-C1(1,2)).^2));
dist1(7) = sum(sqrt((D1(idx1==2,1)-C1(2,1)).^2 + (D1(idx1==2,2)-C1(2,2)).^2));
dist1(8) = sum(sqrt((D1(idx1==3,1)-C1(3,1)).^2 + (D1(idx1==3,2)-C1(3,2)).^2));
dist1(9) = sum(sqrt((D1(idx1==4,1)-C1(4,1)).^2 + (D1(idx1==4,2)-C1(4,2)).^2));

rng(2);
[idx2, C2] = kmeans(D2, k(1), 'Replicates', 20, 'Distance', 'sqeuclidean'); % D1, k=2
subplot(3,2,2);
plot(D2(idx2==1,1), D2(idx2==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D2(idx2==2,1), D2(idx2==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(C2(:,1), C2(:,2), 'kx', 'MarkerSize', 20, 'LineWidth', 5);
legend('Cluster1', 'Cluster2', 'Centroids');
title('Dataset D2: k = 2');
xlabel('Feature 1');
ylabel('Feature 2');
dist2(1) = sum(sqrt((D2(idx2==1,1)-C2(1,1)).^2 + (D2(idx2==1,2)-C2(1,2)).^2));
dist2(2) = sum(sqrt((D2(idx2==2,1)-C2(2,1)).^2 + (D2(idx2==2,2)-C2(2,2)).^2));

rng(2);
[idx2, C2] = kmeans(D2, k(2), 'Replicates', 20, 'Distance', 'sqeuclidean'); % D1, k=3
subplot(3,2,4);
plot(D2(idx2==1,1), D2(idx2==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D2(idx2==2,1), D2(idx2==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(D2(idx2==3,1), D2(idx2==3,2), 'g.', 'MarkerSize', 10); hold on;
plot(C2(:,1), C2(:,2), 'kx', 'MarkerSize', 20, 'LineWidth', 5);
legend('Cluster1', 'Cluster2', 'Cluster3', 'Centroids');
title('Dataset D2: k = 3');
xlabel('Feature 1');
ylabel('Feature 2');
dist2(3) = sum(sqrt((D2(idx2==1,1)-C2(1,1)).^2 + (D2(idx2==1,2)-C2(1,2)).^2));
dist2(4) = sum(sqrt((D2(idx2==2,1)-C2(2,1)).^2 + (D2(idx2==2,2)-C2(2,2)).^2));
dist2(5) = sum(sqrt((D2(idx2==3,1)-C2(3,1)).^2 + (D2(idx2==3,2)-C2(3,2)).^2));

rng(2);
[idx2, C2] = kmeans(D2, k(3), 'Replicates', 20, 'Distance', 'sqeuclidean'); % D1, k=4
subplot(3,2,6);
plot(D2(idx2==1,1), D2(idx2==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D2(idx2==2,1), D2(idx2==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(D2(idx2==3,1), D2(idx2==3,2), 'g.', 'MarkerSize', 10); hold on;
plot(D2(idx2==4,1), D2(idx2==4,2), 'k.', 'MarkerSize', 10); hold on;
plot(C2(:,1), C2(:,2), 'kx', 'MarkerSize', 20, 'LineWidth', 5);
legend('Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Centroids');
title('Dataset D2: k = 4');
xlabel('Feature 1');
ylabel('Feature 2');
dist2(6) = sum(sqrt((D2(idx2==1,1)-C2(1,1)).^2 + (D2(idx2==1,2)-C2(1,2)).^2));
dist2(7) = sum(sqrt((D2(idx2==2,1)-C2(2,1)).^2 + (D2(idx2==2,2)-C2(2,2)).^2));
dist2(8) = sum(sqrt((D2(idx2==3,1)-C2(3,1)).^2 + (D2(idx2==3,2)-C2(3,2)).^2));
dist2(9) = sum(sqrt((D2(idx2==4,1)-C2(4,1)).^2 + (D2(idx2==4,2)-C2(4,2)).^2));