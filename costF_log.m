function [J, grad_J] = costF_log(theta, X, y)
% costF_log Computes the cost function and its gradient for logistic
% regression
% Input arguments:
% theta - the parameters for logistic regression 1./(1+exp(X*theta))
% X - a matrix whose rows are the input feature vectors, where the
% first
% colomn is 1's.
% y - the corresponding labels of the rows in X ( 0 or 1).
% Output arguments:
% J - the cost function
%
% Usage: [J, grad_J] = costF_log(theta, X, y)
%
% Initialization
m = length(y); % number of training examples
J = 0; % Initial cost
grad_J = zeros(size(theta));
% Computing the hypothesis function for theta using sigmoid function
h_theta=sigmoid2020(X*theta);
% please add here a line to compute J
J=1/m*(-y'*log(h_theta)-(1-y)'*log(1-h_theta)); % cost function
% please add here a line to compute grad_J
grad_J=(1/m)*X'*(h_theta-y);
end
