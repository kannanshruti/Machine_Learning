function y_predict = lda_test(x_test, model, classes)
% Description: Returns predicted labels for the test data using LDA
% Inputs :  X_test : test data 
%           model: LDA model with the following variables
%                   model.mu : n_classes * dimensions
%                   model.sigma: dimensions * dimensions  covariance matrix
%                   model.pi : numofClass * 1 vector, Pi(i) = prior probability of class i
% Output :  y_predict: predicted labels for all the testing data points in x_test

    [no_samples, ~] = size(x_test);
    n_classes = numel(classes);
    h = zeros(n_classes, 1);
    y_predict = zeros(no_samples,1);
    for i = 1:no_samples
        for j = 1:n_classes 
            h(j) = 0.5 * ((x_test(i, :) - model.mu(j,:)) * inv(model.sigma)...
                    * (x_test(i, :) - model.mu(j,:))')...
                    + 0.5*log(det(model.sigma)) - log(model.pi(j));
        end
        [~, pred_ind] = min(h); % Finding the index of h where h is minimum, i.e. argmin
        y_predict(i,1) = classes(pred_ind); % Label of the data is the index of the minimum h value
    end
end