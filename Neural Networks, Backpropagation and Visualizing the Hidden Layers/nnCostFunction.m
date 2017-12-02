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


% Part1 
% adding the bias node
a1 = [ones(m,1) X];
a2 = sigmoid(a1 * Theta1');

% again, add another bias node
a2 = [ones(size(a2,1),1) a2];
a3 = sigmoid(a2 * Theta2');

% labels for multiclass classification
y_label = zeros(m, num_labels);

% assign the labels accrodingly
for i = 1:m
  y_label(i, y(i)) = 1;
end;

% implemeting the cost function 

J = 1/m * sum(sum(-y_label .* log(a3)-(1-y_label) .* log(1-a3)));

% here we use sumsq() to perform sum of square
% when performing regularization, we omit the bias node 
regularize = lambda/(2*m) * (sum(sumsq(Theta1(:,2:end))) + sum(sumsq(Theta2(:,2:end))));

J = J + regularize;


% Part2
% Backword Propagation

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for t = 1:m
  % extracting row by row (observation) of the data
  % a1 is 400+1 x 1 matrix (bias node added)
  a1 = [1; X(t,:)'];
  % a2 is the sigmoid result of 1x401 by 401x1 matrix product
  % adding bias node to a2 (hidden layer)
  a2 = [1;sigmoid(Theta1 * a1)];
  
  a3 = sigmoid(Theta2 * a2);
  
  % using a loop to create y_label for multiclass labels
  % initialize y_label (trying another method)
  y_label = zeros(1, num_labels);
  y_label(y(t)) = 1;
  
  % backpropagation starts here by computing delta
  % need to add bias node too for backprop.
  % but remember to exclude during accummulation
  delta3 = a3 - y_label';
  delta2 = Theta2' * delta3 .* [1;sigmoidGradient(Theta1 * a1)];
  
  Theta1_grad = Theta1_grad + delta2(2:end) * a1';
  Theta2_grad = Theta2_grad + delta3 * a2';
  

  
end;


% no need to regularize the bias node
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
