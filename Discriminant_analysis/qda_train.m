function [model] = qda_train(x_train, y_train)
% Description: Returns the QDA model for the dataset
% Input:  x_train: Training data (no_samples*dimensions)
%         y_train: Training labels (no_samples*1)
% Output: model: QDA model with 3 variables
%                 model.mu: mean feature vectors for each class (n_classes*dimensions)
%                 model.sigma: covariance matrices for each class (dimension*dimension*n_classes)
%                 model.pi: prior probability (n_classes*1) (n_samples_class/total_samples)
    [no_samples, ~] = size(x_train);
    classes = unique(y_train);
    n_classes = numel(classes);
    
    for i = 1:n_classes
        current_class = find(y_train == classes(i));
        sum_class = sum(x_train(current_class, :));
        cov_class = x_train(current_class, :);
        model.mu(i,:) = sum_class / numel(current_class);
        model.sigma(:,:,i) = cov(cov_class); 
        model.pi(i) = numel(current_class) / no_samples;
    end
end