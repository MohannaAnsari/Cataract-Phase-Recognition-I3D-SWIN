import torch, numpy as np
from scipy.signal import medfilt
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score
from i3d_model import build_i3d
from dataset_clip import val_ds, val_loader

def run_evaluation(model_path):
    
    model = build_i3d(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.cuda().eval()

    stride = 16
    results, metrics = {}, {}

    for vid in val_ds.data["VideoID"].unique():
        clips, starts = val_ds.get_clips_for_video(vid, clip_len=32, stride=stride)
        preds = []
        for clip in clips:
            with torch.no_grad():
                clip = clip.permute(1, 0, 2, 3)  # [T,C,H,W] → [C,T,H,W]
                p = model(clip.unsqueeze(0).cuda()).softmax(1).cpu().numpy()
            preds.append(p)
        preds = np.concatenate(preds)
        timeline = np.repeat(preds.argmax(1), stride)

        # postprocessing
        timeline_med = medfilt(timeline, kernel_size=7)
        timeline_vote = np.copy(timeline)
        for i in range(len(timeline)):
            start = max(0, i - 5)
            end = min(len(timeline), i + 6)
            timeline_vote[i] = mode(timeline[start:end], keepdims=True)[0][0]

        gt = val_ds.get_ground_truth_for_video(vid)
        gt = gt[:len(timeline)]  # align lengths

        metrics[vid] = {
            "raw_acc": accuracy_score(gt, timeline),
            "med_acc": accuracy_score(gt, timeline_med),
            "vote_acc": accuracy_score(gt, timeline_vote),
            "raw_f1": f1_score(gt, timeline, average="macro"),
            "med_f1": f1_score(gt, timeline_med, average="macro"),
            "vote_f1": f1_score(gt, timeline_vote, average="macro"),
        }

        results[vid] = {"raw": timeline, "median": timeline_med, "vote": timeline_vote}

    avg_metrics = {k: np.mean([m[k] for m in metrics.values()]) for k in metrics[list(metrics.keys())[0]]}
    return avg_metrics, results

# if __name__ == "__main__":
#     run_evaluation("outputs/models/i3d_best.pth")
