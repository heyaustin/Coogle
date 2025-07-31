import torch
from train import CNNKeyword
# from pathlib import Path
# from optimum.exporters.onnx import main_export
import torch
import torch.nn as nn
from train import CNNKeyword, mfcc_transform, SAMPLE_RATE
# from onnx.defs import onnx_opset_version

N_CLASSES = 10
onnx_path = "model.onnx"


class MyWrapper(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.mfcc = mfcc_transform
        self.net = CNNKeyword(n_classes)

        # initialise LazyLinear
        # with torch.inference_mode():
        #     _ = self.forward(torch.zeros(1, SAMPLE_RATE))

    def forward(self, wav: torch.Tensor):
        """
        Args
        ----
        wav: (B, L)   raw PCM, 16‑kHz
        Returns
        -------
        (B, n_classes) unnormalised logits
        """
        x = self.mfcc(wav)          # (B, n_mfcc, T)
        x = x.unsqueeze(1)          # add channel → (B, 1, n_mfcc, T)
        print("Input shape:", x.shape)
        return self.net(x)


model = MyWrapper(N_CLASSES)
model.net.load_state_dict(
    torch.load("model.pt", map_location="cpu")
)
model.eval()

dummy = torch.randn(1, 16000)  # 1 s at 16 kHz
torch.onnx.export(
    model, dummy, "model.onnx",
    opset_version=17,
    input_names=["audio"],
    output_names=["logits"],
    dynamic_axes={
        "audio": {1: "n_samples"},
        # "logits": {1: "n_frames"}
    }
)

# with torch.inference_mode():
#     _ = model(dummy)                       # initialise any Lazy* layers
