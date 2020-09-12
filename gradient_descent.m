function [theta, J,grad_J] = gradient_descent(X, y, theta, alpha, num_iters)
% gd Performs gradient descent to obtain theta
% theta = gd(X, y, theta, alpha, num_iters) updates theta by
% using num_iters gradient steps with learning rate alpha
% Input arguments:
% X a matrix whose rows are the input feature vectors, where the
% first
% column is 1's.
% y - the corresponding labels of the rows in X ( 0 or 1).
% theta - initial theta
% alpha - the learning rate
% num_iters - maximum number of iterations
% Output arguments
% theta - the parameters for logistic regression 1./(1+exp(-X*theta))
% J - cost function J(theta)
% grad - the gradient of the cost function
% Usage: [theta, J,grad] = gradient_descent_2020(X, y, theta, alpha, num_iters)
%
m=length(y);
J_iter=zeros(num_iters,1);
[J, grad_J] = costF_log(theta, X, y);
k=1;
while k<=num_iters
    theta=theta-alpha*grad_J;
    [J, grad_J] = costF_log(theta, X, y);
    J_iter(k)=J;
    k=k+1;
end
J=J_iter;


