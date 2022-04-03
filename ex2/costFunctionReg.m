function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%J = -1/m sumover 1 to m[y' log(h(x)) + (1-y') log(1-h(x))]

h = sigmoid(X*theta);


redtheta = theta;
redtheta(1) = [];

J = -1/m*sum((1.-y)'*log(1.-h)+y'*log(h)) + sum(lambda/(2*m)*redtheta.^2);

grad_zero = 1/m*((h-y)' * X)';
grad_red = (lambda/m)*redtheta;

grad_red = [0;grad_red];
grad = grad_zero + grad_red;




% =============================================================

end
