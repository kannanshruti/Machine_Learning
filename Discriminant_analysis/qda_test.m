function [y_predict] = qda_test(x_test, model, classes)
% Description: Predicts labels for the test data
% Input:  x_test: List of test data (no_samples * dimensions)
%         model: QDA model with variables mu, sigma and pi
%         classes: List of classes
% Output: y_predict: Predicted labels for the test data
    n_classes = numel(classes);
    h = zeros(n_classes,1);
    y_predict = zeros(size(x_test,1));
    for i = 1:size(x_test,1)
        for j = 1:n_classes
            tmp = 0.5 * (x_test(i,:)-model.mu(j,:)) * inv(model.sigma(:,:,j)) * transpose(x_test(i,:)-model.mu(j,:));
            h(j) = tmp + 0.5*log(det(model.sigma(:,:,j))) - log(model.pi(j));
        end
        [~, pred_ind] = min(h);
        y_predict(i) = classes(pred_ind);
    end
end