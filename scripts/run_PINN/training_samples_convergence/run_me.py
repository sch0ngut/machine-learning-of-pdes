import pandas as pd
from termcolor import cprint
from burgers_ml.PINN import PINN

n_training_samples = [40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
result_df = pd.DataFrame(columns=['n', 'epochs', 'e_MAE', 'loss_IC', 'loss_BC', 'loss_train', 'loss_coll', 'e_MSE']).set_index('n')
min_mse = 0.05

for n in n_training_samples:

    # import tensorflow as tf

    cprint(f"%%%% Number of training samples: {n}", 'red')
    pinn = PINN(tf_seed=2)
    pinn.generate_training_data(n_initial=int(n/2), n_boundary=int(n/4))
    pinn.perform_training(min_mse=min_mse)

    losses = pinn.get_losses()
    res = [pinn.epoch, losses[5], losses[0], losses[1], losses[2], losses[3], pinn.mse]
    result_df.loc[n] = res

result_df.to_csv(f'scripts/run_PINN/training_samples_convergence/df_{min_mse}')
print(result_df)



