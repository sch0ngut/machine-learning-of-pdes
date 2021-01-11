from burgers_ml.PINN import PINN

# Part 1: Build and train PINN
pinn = PINN()
pinn.generate_training_data(n_initial=50, n_boundary=25, equidistant=False)
pinn.perform_training(min_mse=0.0005, track_losses=False, batch_size='full')

#