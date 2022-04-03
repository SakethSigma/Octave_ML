function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_trial = 0.01;
sigma_trial = 0.01;


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
%{
values = [0.01, 0.03, 0.1,0.3,1,3,10,30];
values_C = values;
values_sigma = values;
current_lowest_error = Inf;

A = zeros(3,1);

for C_trial = values_C
  for sigma_trial =  values_sigma
    model= svmTrain(X, y, C_trial, @(x1, x2) gaussianKernel(x1, x2, sigma_trial));
    predictions = svmPredict(model,Xval);
    prediction_error = mean(double(predictions ~=yval));
    
    if (prediction_error<current_lowest_error)
      current_lowest_error = prediction_error;
      A(1) = current_lowest_error;
      A(2) = C_trial;
      A(3) = sigma_trial;
    endif

%    fprintf("C, sigma, Error [%f %f %f]\n",C_trial,sigma_trial,prediction_error);
%    fprintf("\n");
    
  end
  
end

error = A(1);
C = A(2);
sigma = A(3);
%}
C = 1;
sigma = 0.1;




% =========================================================================

end
