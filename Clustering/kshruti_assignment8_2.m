data = load('BostonListing');
neigh = data.neighbourhood;
latitude = data.latitude;
longitude = data.longitude;
n = size(neigh, 1);
k = 1:25;
uniq_neigh = unique(neigh);
purity = zeros(length(k),1);
idx = zeros(n, length(k));
ind_neigh = zeros(n, 1);

sig = 0.01;
D1 = [latitude,longitude];


for i = 1:n
    ind = find(strcmp(neigh(i),uniq_neigh));
    ind_neigh(i,1) = ind;
end

%% S, W, L
for i = 1:n
    temp = bsxfun(@minus, D1(i,:),D1); % Row of sample -  All other rows
    S1(i,:) = sum(temp.^2,2)'; % Row corresponds to Sample  
end
S1 = exp(-S1 ./(2*sig^2));
deg1 = diag(sum(S1)); % Col corresponds to Node, sum = sum of weights
L1 = deg1 - S1; % W = S
L1_sym = (deg1)^-0.5 * L1 * (deg1)^-0.5;

%% Purity calculation

[V, G] = svd(L1_sym);
for i = 1:length(k)
    V1 = V(:,end-k(i)+1:end);
    V1 = V1 ./ repmat(sqrt(sum(V1.^2,2)),1,k(i));
    rng(2);
    idx(:,i) = kmeans(V1, k(i)); % SC3, k = 1 to 25
    count = 0;
    for j = 1:k(i)
        idx1 = find(idx(:,i) == j); % Where idx has that particular k
        val1 = ind_neigh(idx1); % Values of ind_neigh corres to those indices
        mode_neigh = mode(val1); % Most frequently occuring 'neighbourhood' in those values
        num_neigh(i,j) = sum(val1 == mode_neigh); % No of occurences of the mode
        count = count + numel(val1);
        if i == 5
            name(j) = uniq_neigh(mode_neigh);
        end
    end
    purity(i) = sum(num_neigh(i,:),2) / count; 
end

%% Plot
figure;
plot(k, purity)
title('k Vs Purity')
xlabel('k')
ylabel('Purity')

%% Plotting data points on Google map
figure;
plot(longitude(idx(:,5)==1),latitude(idx(:,5)==1),'.r','MarkerSize',20); hold on;
plot(longitude(idx(:,5)==2),latitude(idx(:,5)==2),'.b','MarkerSize',20);
plot(longitude(idx(:,5)==3),latitude(idx(:,5)==3),'.g','MarkerSize',20);
plot(longitude(idx(:,5)==4),latitude(idx(:,5)==4),'.k','MarkerSize',20);
plot(longitude(idx(:,5)==5),latitude(idx(:,5)==5),'.m','MarkerSize',20);
plot_google_map
title('Plot of data points on Google map');
legend(name);
xlabel('Longitude');
ylabel('Latitude');