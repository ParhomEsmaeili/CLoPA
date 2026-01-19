from torch.optim import (
    AdamW,
    Adam,
    SGD,
    RMSprop
)
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR,
    ExponentialLR,

)
from nnInteractive.adaptation.training_utils.general_utils import make_factory

optimiser_algo_registry = {
    'adamW': make_factory(AdamW),
    'adam': make_factory(Adam)
}

lr_scheduler_registry = {
    'stepLR': make_factory(StepLR),
    'cosineAnnealingLR': make_factory(CosineAnnealingLR),
    'reduceLROnPlateau': make_factory(ReduceLROnPlateau),
    'linearLR': make_factory(LinearLR),
    'exponentialLR': make_factory(ExponentialLR)
}
