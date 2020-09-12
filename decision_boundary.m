function decision_boundary(X,theta)
x1min=min(X(:,2));
x1max=max(X(:,2));
line([x1min,x1max],...
    [(-theta(1)-theta(2)* x1min)/theta(3),...
    (-theta(1)-theta(2)* x1max)/theta(3)],...
    'LineWidth', 3, 'Color', 'b')
end




