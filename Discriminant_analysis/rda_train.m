function model = rda_train(x_train, y_train, gamma)
    [no_samples, D] = size(x_train);
    classes = unique(y_train);
    n_classes = numel(classes);
    sigma = 0;

    for i = 1:n_classes
        cov_class = zeros(D);
        
        current_class = find(y_train == classes(i)); % Find labels belonging to i-th class
        sum_class = sum(x_train(current_class,:));
        cov_class = cov(x_train(current_class,:));
        
        model.mu(i,:) = sum_class / numel(current_class); % Sum of data of class/ no of samples in class
        model.pi(i) = numel(current_class) / no_samples; % No of samples belonging to class/ Total no of samples
        sigma = sigma + cov_class * model.pi(i); % Covariance of the whole data
    end
    model.sigma = gamma * diag(diag(sigma)) + (1-gamma) * sigma; 
end
