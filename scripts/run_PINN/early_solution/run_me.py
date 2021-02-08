from burgers_ml.PINN import PINN
from util.generate_plots import *

# Initialise network and train
pinn = PINN()
pinn.generate_training_data(n_initial=50, n_boundary=25, equidistant=False)
pinn.perform_training(max_n_epochs=50, min_mse=0.0005, track_losses=True, batch_size='full')

# Plot solution
generate_contour_and_snapshots_plot(pinn.u_pred, train_feat=pinn.train_feat,
                                    savefig_path='scripts/run_PINN/early_solution/contour_and_snapshots_plot.jpg')
generate_contour_plot(pinn.u_pred, train_feat=pinn.train_feat,
                      savefig_path='scripts/run_PINN/early_solution/contour_plot.jpg')

# Plot loss: overall training loss
plot_df1 = pinn.loss_df[['loss_train', 'loss_coll', 'error']]
plot_df1.columns = ['loss on training data', 'loss on collocation points', 'error']
color_dict1 = {
    'loss on training data': 'green',
    'loss on collocation points': 'orange',
    'error': 'red'
}
generate_loss_plot(plot_df1, color_dict=color_dict1, savefig_path='scripts/run_PINN/early_solution/loss_plot1.jpg')

# Plot loss: training loss split in loss on initial and boundary conditions
plot_df2 = pinn.loss_df[['loss_IC', 'loss_BC', 'loss_coll', 'error']]
plot_df2.columns = ['loss on initial data', 'loss on boundary data', 'loss on collocation points', 'error']
color_dict2 = {
    'loss on initial data': 'darkgreen',
    'loss on boundary data': 'lightgreen',
    'loss on collocation points': 'orange',
    'error': 'red'
}
generate_loss_plot(plot_df2, color_dict=color_dict2, savefig_path='scripts/run_PINN/early_solution/loss_plot2.jpg')

pinn.network.save('scripts/run_PINN/early_solution/model.h5')
