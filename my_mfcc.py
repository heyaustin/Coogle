import torch
import torchaudio
from torch import nn


class MFCCFrontend(nn.Module):
    def __init__(
        self, sr=16_000, n_mfcc=40, n_mels=40,
        win_len=400, hop_len=160
    ):
        super().__init__()
        self.win_len, self.hop_len = win_len, hop_len
        self.register_buffer("window", torch.hann_window(win_len))

        # Mel filterbank (constant)
        # mel_fb = torchaudio.functional.melscale_fbanks(
        #     (win_len // 2) + 1,
        #     sr, f_min=20.0, f_max=sr / 2,
        #     n_mels=n_mels, mel_scale="htk"
        # )
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=(win_len // 2) + 1,   # (= n_fft//2 + 1)
            f_min=20.0,
            f_max=sr / 2,
            n_mels=n_mels,
            sample_rate=sr,               # <-- keyword makes intent explicit
            mel_scale="htk"
        )
        self.register_buffer("mel_fb", mel_fb)

        # DCT matrix (constant) for MFCC
        dct = torchaudio.functional.create_dct(n_mfcc, n_mels, norm="ortho")
        self.register_buffer("dct_mat", dct)

    def forward(self, wav):            # wav: (B, N)
        stft = torch.stft(
            wav, n_fft=self.win_len, hop_length=self.hop_len,
            win_length=self.win_len, window=self.window,
            return_complex=False
        )
        power = stft.pow(2).sum(dim=-1)            # (B, 201, T)

        mel_spec = torch.matmul(
            power.transpose(1, 2),                 # (B, T, 201)
            self.mel_fb                            # (201, 40)
        ).transpose(1, 2)                          # (B, 40, T)

        log_mel = torch.log1p(mel_spec)            # (B, 40, T)
        mfcc = torch.matmul(self.dct_mat, log_mel)  # (B, n_mfcc, T)
        return mfcc.transpose(1, 2)                # (B, T, n_mfcc)

    # def forward(self, wav):   # wav: (B, N)
    #     stft = torch.stft(
    #         wav,
    #         n_fft=self.win_len,
    #         hop_length=self.hop_len,
    #         win_length=self.win_len,
    #         window=self.window,
    #         return_complex=False,
    #         onesided=True,
    #         # export_options={"opset_version": 21}
    #     )
    #     power = stft.pow(2).sum(dim=-1)
    #     # power = stft.real.pow(2) + stft.imag.pow(2)

    #     # print("self.mel_fb shape:", self.mel_fb.shape)  # (B, n_freqs, n_frames, 2)
    #     # print("Power shape:", power.shape)  # (B, n_freqs,
    #     mel_spec = torch.matmul(self.mel_fb, power)
    #     log_mel = torch.log(mel_spec + 1.0e-10)
    #     mfcc = torch.matmul(self.dct_mat, log_mel)
    #     return mfcc.transpose(1, 2)  # (B, frames, n_mfcc)
