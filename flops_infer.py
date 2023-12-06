import os
import argparse
from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table
import matplotlib.pylab as plt
import numpy as np
from fvcore.nn import FlopCountAnalysis

import torch
from torch.utils.data import DataLoader

import hparams
from flops_model import Tacotron2
from utils_data import TextMelLoader, TextMelCollate
from utils_public import parse_batch


def plot_data(data, index, path, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect="auto", origin="lower", interpolation="none")
    file = os.path.join(path, str(index) + ".png")
    plt.savefig(file)
    plt.close()


def denormalize_feats(feat, cmvn_path):
    feat = feat.detach().cpu().numpy()
    cmvn = np.load(os.path.join(cmvn_path, "cmvn.npy"))
    mean = cmvn[:, 0:1]
    std = cmvn[:, 1:]
    feat = (feat * std) + mean
    feat = torch.from_numpy(feat)
    return feat


def load_avg_checkpoint(checkpoint_path):
    checkpoint_restore = torch.load(checkpoint_path[0])["state_dict"]
    for idx in range(1, len(checkpoint_path)):
        checkpoint_add = torch.load(checkpoint_path[idx])["state_dict"]
        for key in checkpoint_restore:
            checkpoint_restore[key] = checkpoint_restore[key] + checkpoint_add[key]

    for key in checkpoint_restore:
        if key.split(".")[-1] == "num_batches_tracked":
            checkpoint_restore[key] = checkpoint_restore[key] // (len(checkpoint_path))
        else:
            checkpoint_restore[key] = checkpoint_restore[key] / (len(checkpoint_path))
    return checkpoint_restore


def main(args, hparams):

    # prepare data
    testset = TextMelLoader(hparams.test_files, hparams, shuffle=False)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    test_loader = DataLoader(
        testset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # prepare model
    model = Tacotron2(hparams).cuda(device="cuda:0")
    checkpoint_restore = load_avg_checkpoint(args.checkpoint_path)
    model.load_state_dict(checkpoint_restore)
    model.eval()
    print("# total parameters:", sum(p.numel() for p in model.parameters()))

    # infer for flops statistics
    for batch in test_loader:
        input, _ = parse_batch(batch)
        break
    text_inputs, _, mels, _, _ = input
    output, _ = model(text_inputs, mels)

    flops = FlopCountAnalysis(model, (text_inputs, mels))
    print("-------------------------STRING-----------------------------")
    print(flop_count_str(flops))
    print("-------------------------TABLE------------------------------")
    print(flop_count_table(flops))
    mel_predict = output[0]
    mel_denorm = denormalize_feats(mel_predict, hparams.dump)
    mel_path = os.path.join(args.output_infer, "{:0>3d}".format(0) + ".pt")
    torch.save(mel_denorm, mel_path)
    plot_data(
        (
            mel_predict.detach().cpu().numpy(),
            mel_denorm.numpy(),
        ),
        0,
        args.output_infer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_infer",
        type=str,
        default="output_infer",
        help="directory to save infer outputs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=[
            "/home/server/disk1/checkpoints/tacotron2_tts/baseline/checkpoint_200000",
            "/home/server/disk1/checkpoints/tacotron2_tts/baseline/checkpoint_199000",
            "/home/server/disk1/checkpoints/tacotron2_tts/baseline/checkpoint_198000",
            "/home/server/disk1/checkpoints/tacotron2_tts/baseline/checkpoint_197000",
            "/home/server/disk1/checkpoints/tacotron2_tts/baseline/checkpoint_196000",
        ],
        required=False,
        help="checkpoint path for infer model",
    )
    args = parser.parse_args()
    os.makedirs(args.output_infer, exist_ok=True)
    assert args.checkpoint_path is not None

    main(args, hparams)
    print("finished")
