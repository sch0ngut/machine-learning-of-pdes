from burgers_ml.PINN import PINN

pinn = PINN()
pinn.generate_training_data(n_initial=50, n_boundary=25, equidistant=False)