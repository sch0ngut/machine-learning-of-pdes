function plot_burgers(solution, x, t)
% Plots the solution to the Burger's equation
[T,X] = meshgrid(t,x);
surface(T,X,solution)
axis tight
colormap jet
shading interp
colorbar
xlabel('t');
ylabel('x');
set(gca,'FontSize',14);
set(gcf, 'Color', 'w');

%end

