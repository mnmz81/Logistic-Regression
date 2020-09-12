function [theta,J_cost,grad_J]=gd_reg(X,y,theta,alpha,num_iters,lamda)
    m=length(y);
    J_iter=zeros(num_iters,1);
    [J_cost, grad_J] = costF_reg(theta, X, y,lamda);
    k=1;
    while k<=num_iters
        theta=theta-alpha*grad_J;
        [J_cost, grad_J] = costF_reg(theta,X,y,lamda);
        J_iter(k)=J_cost;
        k=k+1;
    end
    J_cost=J_iter;
end
