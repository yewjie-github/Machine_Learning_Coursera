function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


% essentially, the hypothesis for logistic regression is 
% a sigmoid function
hypo = sigmoid(X * theta);

% Cost function
% please take note that you need to tranpose y (i.e. y')
% in order to perform matrix arithmetic operation
J = (1/m) * sum(-y' * log(hypo) - (1 - y') * log(1 - hypo));

% Gradient Descent
grad = (1/m) * (X' *(hypo - y));





% =============================================================

end
