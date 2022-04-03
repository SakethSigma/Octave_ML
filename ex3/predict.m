function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(size(X,1),1),X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
A2 = sigmoid(X*Theta1');
%X is 5000 x 401 matrix. Theta1 is 25 x 401 (size of next layer x size of current layer+1)
%a2 will be 5000 x 25 matrix where parameters have been reduced to 25 from 400.
A2 = [ones(size(A2,1),1),A2];
%a2 is 5000 x 26 matrix. Theta2 is 10x26 Matrix. 
A3 = sigmoid(A2*Theta2');
%Theta2' is 26x10 matrix. a3 is 5000 x 10 matrix where 5000 are training examples.
%the columns are probability that each classifier(number is true).
%Index of max of probability in each row will give the correct prediction.
[val,ind] = max(A3,[],2);
p = ind;







% =========================================================================


end
