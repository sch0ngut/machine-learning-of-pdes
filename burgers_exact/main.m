clear

N_x_vec = [161, 321, 641, 1281, 2561, 5121];
N_t_vec = [10^3+1, 10^4+1];


for N_t = N_t_vec
    for N_x = N_x_vec 
        
        fprintf(2, 'N_t=%g, N_x=%g \n', N_t, N_x);
        
        x = linspace(-1,1,N_x);
        t = linspace(0, 1, N_t);
        mysol= burgers_viscous_time_exact1(0.01/pi, N_x, x, N_t, t);
        
        figure
        plot_burgers(mysol, x, t)
        title(['N_x=' num2str(N_x) ', N_t=' num2str(N_t)]);
%         saveas(gcf, join([save_figs_path, 'N_t=' num2str(N_t), '_N_x=', num2str(N_x),'_.jpg'], ""))
%         close(gcf)
        save(join(['solutions/burgers_exact_N_t=' num2str(N_t), '_N_x=', num2str(N_x)], ""), 'mysol')
    end
end
