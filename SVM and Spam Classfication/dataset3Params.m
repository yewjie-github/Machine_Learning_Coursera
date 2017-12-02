function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_sigma_value = [0.01 0.03 0.1 0.3 1 3 10 30];

% initializing variable to store result
% c_sigma_value ^ 2 represents the number of possible results
% the three columns represent the C, sigma and error values respectively
results = zeros((length(c_sigma_value) ^ 2), 3);

% looping variable for results
result_row = 0;

for i = 1:length(c_sigma_value)
  for j = 1:length(c_sigma_value)
    % increment of result_row
    result_row = result_row + 1 ;
    
    % model the training algo with different C and sigma value
    model = svmTrain(X, y, c_sigma_value(i), @(x1, x2) gaussianKernel(x1, x2, c_sigma_value(j)));
    % using the model to make predictions
    predictions = svmPredict(model, Xval);
    % evaluating the error of each prediction
    prediction_error = mean(double(predictions ~= yval));
    
    % store the C, sigma, and prediction_error accordingly into the results variable
    results(result_row, :) = [c_sigma_value(i), c_sigma_value(j), prediction_error];
  end
  
end

% since we are only interested with the minimal error
% we sort only the result wrt to the prediction_error (third column)
results_sorted = sortrows(results, 3);

% then we extract the C and sigma value for the row with the
% minimal prediction_error

C = results_sorted(1,1);
sigma = results_sorted(1,2);

% =========================================================================

end
