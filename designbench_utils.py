import torch
import matplotlib.pyplot as plt
import numpy as np
import design_bench
from torch.utils.data import TensorDataset, DataLoader


from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
from design_bench.datasets.discrete.gfp_dataset import GFPDataset
from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import (
    DKittyMorphologyDataset,
)
from design_bench.datasets.continuous.hopper_controller_dataset import (
    HopperControllerDataset,
)
from design_bench.datasets.discrete.utr_dataset import UTRDataset
from design_bench.datasets.continuous.superconductor_dataset import (
    SuperconductorDataset,
)

def get_task_and_dataset(task_name, map_to_logits=True):
    """This method returns the offline dataset and the task object 
    for a given task. The task object is required to call the 
    oracle for evaluating candidate designs.
    """
    if task_name == "tfbind8":
        task = design_bench.make("TFBind8-Exact-v0")
        task_dataset = TFBind8Dataset()
    elif task_name == "tfbind10":
        task = design_bench.make("TFBind10-Exact-v0")
        task_dataset = TFBind10Dataset()
    elif task_name == "gfp":
        task = design_bench.make("GFP-Transformer-v0")
        task_dataset = GFPDataset()
    elif task_name == "chembl":
        task = design_bench.make(
            f"ChEMBL_MCHC_CHEMBL3885882" f"_MorganFingerprint-RandomForest-v0"
        )
        task_dataset = ChEMBLDataset(
            assay_chembl_id="CHEMBL3885882", standard_type="MCHC"
        )
    elif task_name == "utr":
        task = design_bench.make("UTR-ResNet-v0")
        task_dataset = UTRDataset()
    elif task_name == "superconductor":
        task = design_bench.make(f"Superconductor-RandomForest-v0")
        task_dataset = SuperconductorDataset()
    elif task_name == "ant":
        task = design_bench.make(f"AntMorphology-Exact-v0")
        task_dataset = AntMorphologyDataset()
    elif task_name == "dkitty":
        task = design_bench.make(f"DKittyMorphology-Exact-v0")
        task_dataset = DKittyMorphologyDataset()
    elif task_name == "hopper":
        task = design_bench.make(f"HopperController-Exact-v0")
        task_dataset = HopperControllerDataset()
    if task_name in ["tfbind8", "tfbind10", "gfp", "utr", "chembl"] and map_to_logits is True:
        print(task.x.shape)
        task.map_to_logits()
        task_dataset.map_to_logits()
        print(task.x.shape)
    return (
        task,
        task_dataset,
    ) 

def get_train_valid_dataloaders(
    x,
    y,
    val_size=200,
    batch_size=128,
    seed=0,
):
    """This method splits the offline dataset into training and 
    validation datasets. It returns a pytorch dataloader that 
    iterates over the training/validation sets.
    """
    # shuffle the dataset using a common set of indices
    torch.manual_seed(seed)
    np.random.seed(seed)
    indices = torch.randperm(x.shape[0])

    # create a training and validation split
    x = x[indices]
    y = y[indices]
    train_inputs = [x[val_size:], y[val_size:]]
    validate_inputs = [x[:val_size], y[:val_size]]
    size = x.shape[0] - val_size

    # build the parallel PyTorch data loading pipeline
    training_dataset = TensorDataset(*train_inputs)
    validation_dataset = TensorDataset(*validate_inputs)
    training_dataset = DataLoader(training_dataset, shuffle=True, batch_size=batch_size)
    validation_dataset = DataLoader(validation_dataset, batch_size=batch_size)

    return (training_dataset, validation_dataset)
