function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);


% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%a1 = [ones(m,1),X];
%z2 = a1*Theta1';
%a2 = sigmoid(z2);
%a2 = [ones(size(a2,1),1),a2];
%hx = sigmoid(a2 * Theta2')';
%Y = zeros(m,num_labels);


%for i = 1:m
%  Y(i,y(i)) = 1;
%end  

%size(Y,1)
%size(Y,2)
%size(hx,1)
%size(hx,2)

%hx should is 5000 x 10 matrix
%J = sum(sum(-1/m*(log(hx).*Y'+log(1-hx).*(1-Y'))));


for i = 1:m
  
  
  a1 = [1,X(i,:)];
  %a1 is 1x401 matrix
  z2 = a1*Theta1';
  % Theta1' is 401 x 25
  %z2 is  1 x 25
  a2 = sigmoid(z2);

  a2 = [1,a2];
  %a2 is 1 x 26

  hx = sigmoid(a2 * Theta2')';
  % Theta2' is 26 x 10
  % Theta2 is 10 x 26
  % hx is 1 x 10

 
  Y = zeros(1,num_labels);
  %creates vector Y with 1xnum_labels dimension

  Y(1,y(i)) = 1;
  %sets ith column as 1.
  

  %hx should is 5000 x 10 matrix
  J = J - sum(1/m*(log(hx).*Y'+log(1-hx).*(1-Y')));
  % cost of that training example
  
  
  %used to compute partial derivate of Cost with respect to each theta parameter in each layer.
  %Capitak_delta will be used as a accumulator.
  delta3 = (hx - Y'); %10x1
  
  delta2 = Theta2'*delta3 .* a2' .* (1-a2'); %26x1
  delta2 = delta2(2:end); %25x1
  
  Theta2_grad = Theta2_grad + delta3*a2; %10x1 * 1 x26 = 10 x 26
  Theta1_grad = Theta1_grad + delta2*a1; %26x1 * 1x401 = 25 x 401
  
%Regularized term cost is each theta element squared.
%All theta terms are in Theta1 and Theta2. Square of their elements and sum of it.
%Then sum of the thetas.
  
end

J_reg = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
J = J + J_reg;

Theta1_grad_reg = [zeros(size(Theta1,1),1),(lambda/m).*Theta1(:,(2:end))];
Theta2_grad_reg = [zeros(size(Theta2,1),1),(lambda/m).*Theta2(:,(2:end))];


Theta1_grad = 1/m.*Theta1_grad + Theta1_grad_reg;
Theta2_grad = 1/m.*Theta2_grad + Theta2_grad_reg;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
