function [J,grad] = costF_reg(theta,X,y,lamda)
    m = length(y); % number of training examples
    h_theta=sigmoid2020(X*theta);
    temp=1/m;
    % please add here a line to compute J
    J=(temp)*((-y)'*log(h_theta)-(1-y)'*log(1-h_theta))+(lamda/(2*m))*(theta'*theta); % cost function
    grad=(temp)*(X'*(h_theta- y))+((lamda./m)*theta);
    grad(1)=grad(1)-((lamda./m)*theta(1));
end
