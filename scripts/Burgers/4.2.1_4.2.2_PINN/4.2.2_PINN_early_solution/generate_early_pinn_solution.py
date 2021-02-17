from machine_learning_solver.PINN import BurgersPINN
from util.generate_plots import *

# Initialise network and train
pinn = BurgersPINN()
pinn.generate_training_data(n_initial=50, n_boundary=25, equidistant=False)
pinn.perform_training(max_n_epochs=50, track_losses=True, batch_size='full')

# Plot solution
generate_contour_plot(pinn.u_pred, train_feat=pinn.train_feat)
# generate_contour_plot(pinn.u_pred, train_feat=pinn.train_feat,
#                       savefig_path='scripts/Burgers/run_PINN/4.2.2_PINN_early_solution/Fig6_contour_plot.jpg')
