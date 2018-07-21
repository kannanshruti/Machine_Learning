% Description: Discriminant analysis
clc; clear; close all;
%% Loading data
load('../files_for_assignment3/data_iris.mat')
[no_samples, D] = size(X); % #Rows is samples, #Cols is dimensions
classes = unique(Y);
no_classes = numel(classes); % #Unique labels equal number of classes
no_iter = 10;
no_train = 100;
temp = 1:no_samples;
CCR = zeros(no_iter, 2);
%%
for i = 1:no_iter
    ind_train = randperm(no_samples, no_train);
    ind_test = temp(~ismember(1:no_samples, ind_train));
    x_train = X(ind_train, :);
    x_test = X(ind_test, :);
    y_train = Y(ind_train, :);
    y_test = Y(ind_test, :);

    QDA_model = qda_train(x_train, y_train);
    y_predQ = qda_test(x_test, QDA_model, classes);

    LDA_model = lda_train(x_train, y_train);
    y_predL = lda_test(x_test, LDA_model, classes);
    
    CCR(i,1) = (numel(find(y_predQ == y_test)) / numel(y_test));
    CCR(i,2) = (numel(find(y_predL == y_test)) / numel(y_test));
end