"""Explicit offline arithmetic diagnostic; never selects or promotes a policy."""
import copy
import torch


class Float64PredictionDiagnostic(torch.nn.Module):
    """Evaluate an independent double-precision copy, preserving API dtype.

    Input values are promoted exactly; checkpoint parameters and caller modules
    remain unchanged. Casts preserve differentiation for diagnostic gradients.
    """
    def __init__(self, predictor):
        super().__init__()
        self.predictor = copy.deepcopy(predictor).double().eval()

    def forward(self, state, action, context):
        with torch.autocast(device_type=state.device.type, enabled=False):
            output = self.predictor(state.double(), action.double(), context.double())
        return output.to(state.dtype)
