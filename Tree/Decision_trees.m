% Description: Regression based decision trees
housing = load('housing_data.mat');
leaf_range = 25;
ae_test = zeros(length(housing.Xtest),leaf_range); 
test = zeros(length(housing.Xtest),leaf_range);
ae_train = zeros(length(housing.Xtrain),leaf_range);
train = zeros(length(housing.Xtrain),leaf_range);

%% Creating the tree using the training data
tree = fitrtree(housing.Xtrain, housing.ytrain,'MinLeafSize',20,...
                'PredictorNames',housing.feature_names);
view(tree, 'Mode', 'graph')

%% Predictions on the test data
sample_medv = predict(tree, [5,18,2.31,1,0.5440,2,64,3.7,1,300,15,390,10]);
test1 = predict(tree, housing.Xtest); 
mae1 = abs(test1 - housing.ytest);

%% Performance metrics
for i = 1:leaf_range 
    tree1 = fitrtree(housing.Xtrain, housing.ytrain,'MinLeafSize',i,...
                'PredictorNames',housing.feature_names);
    test(:,i) = predict(tree1, housing.Xtest); % Predictions for Test
    ae_test(:,i) = abs(test(:,i) - housing.ytest); % Absolute Error: Test
    
    train(:,i) = predict(tree1, housing.Xtrain); % Predictions for Train
    ae_train(:,i) = abs(train(:,i) - housing.ytrain); % Absolute Error: Train
end 

mae_test = mean(ae_test,1); % ANS 1.c
mae_train = mean(ae_train,1); % ANS 1.c

figure;
plot(1:leaf_range, mae_train); hold on;
plot(1:leaf_range, mae_test);
legend('MAE Train', 'MAE Test');
title('Minimum Absolute error (MAE) Vs Minimum Observations per leaf (1-25)');
xlabel('No of observations per Leaf');
ylabel('MAE');