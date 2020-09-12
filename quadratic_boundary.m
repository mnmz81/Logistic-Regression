function quadratic_boundary(X,theta)
    x_axis = linspace(min(X(:,2)),max(X(:,2)),(length(X*theta)));
    plot(x_axis,(-theta(1)-theta(2)*x_axis-theta(4)*x_axis.^2)./theta(3),'LineWidth', 2, 'Color', 'b')
end

