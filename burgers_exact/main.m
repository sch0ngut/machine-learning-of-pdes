clear

n_spatial_vec = [161, 321, 641, 1281, 2561, 5121];
n_temporal_vec = [10^3+1, 10^4+1];

for n_temporal = n_temporal_vec
    for n_spatial = n_spatial_vec 
        
        fprintf(2, 'N_t=%g, N_x=%g \n', n_temporal, n_spatial);
        
        x = linspace(-1,1,n_spatial);
        t = linspace(0, 1, n_temporal);
        u = burgers_viscous_time_exact1(0.01/pi, n_spatial, x, n_temporal, t);
        
        % Plot
%         figure
%         plot_burgers(u, x, t)
%         title(['H=' num2str(n_spatial-1) ', K=' num2str(n_temporal-1)]);
%         saveas(gcf, join([save_figs_path, 'K=' num2str(n_temporal-1), '_H=', num2str(n_spatial-1),'_.jpg'], ""))
%         close(gcf)
        save(join(['solutions/burgers_exact_K=' num2str(n_temporal-1), '_H=', num2str(n_spatial-1)], ""), 'x', 't', 'u')
    end
end
