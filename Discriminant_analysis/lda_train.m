function model = lda_train(x_train, y_train)
% Inputs :  x_train : training data matrix, each row is a training data point
%           y_train : training labels for rows of X_train
% Output :  model : LDAmodel with the following variables
%                 model.mu : numofClass * dimensions
%                 model.sigma : dimensions * dimensions  covariance matrix
%                 LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

    [no_samples, D] = size(x_train);
    classes = unique(y_train);
    n_classes = numel(classes);
    model.sigma = 0;

    for i = 1:n_classes
        cov_class = zeros(D);
        current_class = find(y_train == classes(i)); % Find labels belonging to i-th class
        sum_class = sum(x_train(current_class,:));
        cov_class = cov(x_train(current_class,:));
        
        model.mu(i,:) = sum_class / numel(current_class); % Sum of data of class/ no of samples in class
        model.pi(i) = numel(current_class) / no_samples; % No of samples belonging to class/ Total no of samples
        model.sigma = model.sigma + cov_class * model.pi(i); % Covariance of the whole data
    end
end
