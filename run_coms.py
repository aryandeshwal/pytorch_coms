import torch
import numpy as np

from pytorch_nets import ForwardModel
from pytorch_coms import ConservativeObjectiveModel
from designbench_utils import get_task_and_dataset, get_train_valid_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {"dtype": torch.float64, "device": device}
torch.set_default_dtype(torch.float64)


def main(task_name:str, map_to_logits:bool = False):
    task, task_dataset = get_task_and_dataset(
        task_name=task_name, map_to_logits=map_to_logits
    )
    task.map_normalize_y()
    task.map_normalize_x()
    # offline dataset inputs 
    x = task.x.reshape((-1, np.prod(task.x.shape[1:])))
    # offline dataset outputs
    y = task.y
    x = torch.tensor(x, **tkwargs)
    y = torch.tensor(y, **tkwargs)  
    print(f"x {x.shape}")
    print(f"y {y.shape}")

    train_data_loader, validate_data_loader = get_train_valid_dataloaders(
        x=x, y=y, batch_size=128, val_size=500
    )

    forward_model = ForwardModel(
        x.shape[-1],
        activations=("relu", "relu"),
        hidden_size=2048,
        final_tanh=False,
    )
    forward_model = forward_model.to(device)

    if task_name in ["tfbind8", "tfbind10"]:
        particle_lr = 2.0 
        overestimation_limit = 2.0
    else:
        particle_lr = 0.05 
        overestimation_limit = 0.5  

    trainer = ConservativeObjectiveModel(
        forward_model,
        forward_model_lr=0.0003,
        alpha=0.1,
        alpha_lr=0.01,
        overestimation_limit=overestimation_limit,
        particle_lr=particle_lr,
        entropy_coefficient=0.0,
        noise_std=0.0,
    )

    alphas = trainer.launch(train_data_loader, validate_data_loader, epochs=50)
    print(alphas)
    torch.save(
        {"forward_model": forward_model.state_dict(), "alphas": alphas},
        f"{task_name}_coms_forward_model.pkl",
    )

    # start the search from top k points
    n_evals = 128
    top_idxs = y.squeeze().sort()[1][-n_evals:]
    all_outs = []
    input_shape = task.x.shape[1:]
    for idx in top_idxs:
        initial_score = task.predict(
            x[idx].view([-1, *input_shape]).detach().cpu().numpy()
        ).item()
        initial_score = (initial_score - task_dataset.y.min()) / (
            task_dataset.y.max() - task_dataset.y.min()
        )
        print(f"search's starting point score {initial_score}")
        final_x = trainer.optimize(x[idx], 50)
        score = task.predict(
            final_x.view([-1, *input_shape]).detach().cpu().numpy()
        ).item()
        all_outs.append(
            (task.denormalize_y(score)[0][0] - task_dataset.y.min())
            / (task_dataset.y.max() - task_dataset.y.min())
        )
    print(f"100 percentile {np.sort(all_outs)[int(1 * n_evals)-1]}")
    print(f"50 percentile {np.sort(all_outs)[int(0.5 * n_evals)-1]}")


if __name__ == "__main__":
    # choices for task_names:
    # ["tfbind8", "tfbind10", "gfp", "utr", \
    #  "chembl", "hopper", "ant", "dkitty", "superconductor"]
    main(task_name="ant", map_to_logits=False)
