% Description: Spectral clustering
%% Creating Clusters
D1 = sample_circle(3, [500, 500, 500]);
D2 = sample_spiral(3, [500, 500, 500]);
n = size(D1,1);
rng(2);
sig = 0.2;

%% Calculation of S, W, L
for i = 1:n
    temp = bsxfun(@minus, D1(i,:),D1); % Row of sample -  All other rows
    S1(i,:) = sum(temp.^2,2)'; % Row corresponds to Sample
    temp = bsxfun(@minus, D2(i,:),D2);
    S2(i,:) = sum(temp.^2,2)';        
end
S1 = exp(-S1 ./(2*sig^2));
S2 = exp(-S2 ./(2*sig^2));

deg1 = diag(sum(S1)); % Col corresponds to Node, sum = sum of weights
deg2 = diag(sum(S2));

L1 = deg1 - S1; % W = S; SC-1
L2 = deg2 - S2; % W = Sl; SC-1
L1_rw = deg1\ L1; % SC-2
L2_rw = deg2\ L2; % SC-2
L1_sym = (deg1)^-0.5 * L1 * (deg1)^-0.5; % SC-3
L2_sym = (deg2)^-0.5 * L2 * (deg2)^-0.5; % SC-3

eig_1(:,1) = sort(eig(L1));
eig_1(:,2) = sort(eig(L1_rw));
eig_1(:,3) = sort(eig(L1_sym));
eig_2(:,1) = sort(eig(L2));
eig_2(:,2) = sort(eig(L2_rw));
eig_2(:,3) = sort(eig(L2_sym));

%% kMeans to the Data

[Va, G] = svd(L1_sym); % Eigen Vectors

V1a = Va(:,end-1:end); % D1 k = 2
V1a = V1a ./ repmat(sqrt(sum(V1a.^2,2)),1,2);
rng(2);
figure;
subplot(3,2,1);
[idx1, C1] = kmeans(V1a, 2);
plot(D1(idx1==1,1), D1(idx1==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D1(idx1==2,1), D1(idx1==2,2), 'b.', 'MarkerSize', 10);
legend('Cluster1', 'Cluster2');
title('D1, k = 2');
xlabel('Feature 1');
ylabel('Feature 2');

subplot(3,2,3);
V2a = Va(:,end-3+1:end); % D1 k = 3
V2a = V2a ./ repmat(sqrt(sum(V2a.^2,2)),1,3);
rng(2);
[idx2, C2] = kmeans(V2a, 3);
plot(D1(idx2==1,1), D1(idx2==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D1(idx2==2,1), D1(idx2==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(D1(idx2==3,1), D1(idx2==3,2), 'g.', 'MarkerSize', 10);
legend('Cluster1', 'Cluster2', 'Cluster3');
title('D1, k = 3');
xlabel('Feature 1');
ylabel('Feature 2');

subplot(3,2,5);
V3a = Va(:,end-3:end); % D1 k = 4
V3a = V3a ./ repmat(sqrt(sum(V3a.^2,2)),1,4);
rng(2);
[idx3, C] = kmeans(V3a, 4);
plot(D1(idx3==1,1), D1(idx3==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D1(idx3==2,1), D1(idx3==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(D1(idx3==3,1), D1(idx3==3,2), 'g.', 'MarkerSize', 10); hold on;
plot(D1(idx3==4,1), D1(idx3==4,2), 'k.', 'MarkerSize', 10);
legend('Cluster1', 'Cluster2', 'Cluster3', 'Cluster4');
title('D1, k = 4');
xlabel('Feature 1');
ylabel('Feature 2');

[Vb, G] = svd(L2_sym);
subplot(3,2,2);
V1b = Vb(:,end-1:end); % D2 k = 2
V1b = V1b ./ repmat(sqrt(sum(V1b.^2,2)),1,2);
rng(2);
[idx1, C1] = kmeans(V1b, 2);
plot(D2(idx1==1,1), D2(idx1==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D2(idx1==2,1), D2(idx1==2,2), 'b.', 'MarkerSize', 10);
legend('Cluster1', 'Cluster2');
title('D2, k = 2');
xlabel('Feature 1');
ylabel('Feature 2');

subplot(3,2,4);
V2b = Vb(:,end-2:end); % D2 k = 3
V2b = V2b ./ repmat(sqrt(sum(V2b.^2,2)),1,3);
rng(2);
[idx2, C2] = kmeans(V2b, 3);
plot(D2(idx2==1,1), D2(idx2==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D2(idx2==2,1), D2(idx2==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(D2(idx2==3,1), D2(idx2==3,2), 'g.', 'MarkerSize', 10);
legend('Cluster1', 'Cluster2', 'Cluster3');
title('D2, k = 3');
xlabel('Feature 1');
ylabel('Feature 2');

subplot(3,2,6);
V3b = Vb(:,end-3:end);
V3b = V3b ./ repmat(sqrt(sum(V3b.^2,2)),1,4);
rng(2);
[idx3, C] = kmeans(V3b, 4); % D2 k = 4
plot(D2(idx3==1,1), D2(idx3==1,2), 'r.', 'MarkerSize', 10); hold on;
plot(D2(idx3==2,1), D2(idx3==2,2), 'b.', 'MarkerSize', 10); hold on;
plot(D2(idx3==3,1), D2(idx3==3,2), 'g.', 'MarkerSize', 10); hold on;
plot(D2(idx3==4,1), D2(idx3==4,2), 'k.', 'MarkerSize', 10);
legend('Cluster1', 'Cluster2', 'Cluster3', 'Cluster4');
title('D2, k = 4');
xlabel('Feature 1');
ylabel('Feature 2');

%% Plotting the Eigen vectors of the Laplacian

V2a; % D1 k = 3 SC-3
V2b; % D2 k = 3 SC-3
rng(2);
[idx, C] = kmeans(V2a, 3);
figure;
subplot(3,2,1);
plot3(V2a(idx==1,1), V2a(idx==1,2), V2a(idx==1,3),'rx', 'MarkerSize', 10); hold on;
plot3(V2a(idx==2,1), V2a(idx==2,2), V2a(idx==2,3),'bx', 'MarkerSize', 10); hold on;
plot3(V2a(idx==3,1), V2a(idx==3,2), V2a(idx==3,3),'gx', 'MarkerSize', 10); 
title('SC-3 D1');
legend('Cluster1', 'Cluster2', 'Cluster3'); 
xlabel('V 1');
ylabel('V 2');
zlabel('V 3');
subplot(3,2,2);
plot3(V2b(idx==1,1), V2b(idx==1,2), V2b(idx==1,3),'rx', 'MarkerSize', 10); hold on;
plot3(V2b(idx==2,1), V2b(idx==2,2), V2b(idx==2,3),'bx', 'MarkerSize', 10); hold on;
plot3(V2b(idx==3,1), V2b(idx==3,2), V2b(idx==3,3),'gx', 'MarkerSize', 10); 
title('SC-3 D2');
legend('Cluster1', 'Cluster2', 'Cluster3');
xlabel('V 1');
ylabel('V 2');
zlabel('V 3');

[V1_rw, G] = svd(L1_rw); % D1 k = 3 SC-2
[V2_rw, G] = svd(L2_rw); % D2 k = 3 SC-2
V1_rw = V1_rw(:,end-2:end);
V2_rw = V2_rw(:,end-2:end);
subplot(3,2,3);
plot3(V1_rw(idx==1,1), V1_rw(idx==1,2), V1_rw(idx==1,3),'rx', 'MarkerSize', 10); hold on;
plot3(V1_rw(idx==2,1), V1_rw(idx==2,2), V1_rw(idx==2,3),'bx', 'MarkerSize', 10); hold on;
plot3(V1_rw(idx==3,1), V1_rw(idx==3,2), V1_rw(idx==3,3),'gx', 'MarkerSize', 10); 
title('SC-2 D1');
legend('Cluster1', 'Cluster2', 'Cluster3');
xlabel('V 1');
ylabel('V 2');
zlabel('V 3');
subplot(3,2,4);
plot3(V2_rw(idx==1,1), V2_rw(idx==1,2), V2_rw(idx==1,3),'rx', 'MarkerSize', 10); hold on;
plot3(V2_rw(idx==2,1), V2_rw(idx==2,2), V2_rw(idx==2,3),'bx', 'MarkerSize', 10); hold on;
plot3(V2_rw(idx==3,1), V2_rw(idx==3,2), V2_rw(idx==3,3),'gx', 'MarkerSize', 10); 
title('SC-2 D2');
legend('Cluster1', 'Cluster2', 'Cluster3');
xlabel('V 1');
ylabel('V 2');
zlabel('V 3');

[V1_un, G] = svd(L1); % D1 k = 3 SC-1
[V2_un, G] = svd(L2); % D2 k = 3 SC-1
V1_un = V1_un(:,end-2:end);
V2_un = V2_un(:,end-2:end);
subplot(3,2,5);
plot3(V1_un(idx==1,1), V1_un(idx==1,2), V1_un(idx==1,3),'rx', 'MarkerSize', 10); hold on;
plot3(V1_un(idx==2,1), V1_un(idx==2,2), V1_un(idx==2,3),'bx', 'MarkerSize', 10); hold on;
plot3(V1_un(idx==3,1), V1_un(idx==3,2), V1_un(idx==3,3),'gx', 'MarkerSize', 10); 
title('SC-1 D1');
legend('Cluster1', 'Cluster2', 'Cluster3');
xlabel('V 1');
ylabel('V 2');
zlabel('V 3');
subplot(3,2,6);
plot3(V2_un(idx==1,1), V2_un(idx==1,2), V2_un(idx==1,3),'rx', 'MarkerSize', 10); hold on;
plot3(V2_un(idx==2,1), V2_un(idx==2,2), V2_un(idx==2,3),'bx', 'MarkerSize', 10); hold on;
plot3(V2_un(idx==3,1), V2_un(idx==3,2), V2_un(idx==3,3),'gx', 'MarkerSize', 10); 
title('SC-1 D1');
legend('Cluster1', 'Cluster2', 'Cluster3');
xlabel('V 1');
ylabel('V 2');
zlabel('V 3');

%% Plotting the Eigen values of the Laplacian
figure;
subplot(3,2,1);
plot(1:1500, eig_1(:,1));
title('Dataset D1: Eigen val of L1');
xlabel('n');
ylabel('Eigen Value');
subplot(3,2,3);
plot(1:1500, eig_1(:,2));
title('Dataset D1: Eigen val of L1 rw');
xlabel('n');
ylabel('Eigen Value');
subplot(3,2,5);
plot(1:1500, eig_1(:,3));
title('Dataset D1: Eigen val of L1 sym');
xlabel('n');
ylabel('Eigen Value');

subplot(3,2,2);
plot(1:1500, eig_2(:,1));
title('Dataset D2: Eigen val of L2');
xlabel('n');
ylabel('Eigen Value');
subplot(3,2,4);
plot(1:1500, eig_2(:,2));
title('Dataset D2: Eigen val of L2 rw');
xlabel('n');
ylabel('Eigen Value');
subplot(3,2,6);
plot(1:1500, eig_2(:,3));
title('Dataset D2: Eigen val of L2 sym');
xlabel('n');
ylabel('Eigen Value');