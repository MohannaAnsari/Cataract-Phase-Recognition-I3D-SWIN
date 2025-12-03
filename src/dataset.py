import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml


class CataractDataset(Dataset):
    # Unified dataset for Cataract-101 videos.
    # Can return either:
    #   - clip (video segment) for classification
    #   - frame sequence + labels for segmentation
    
    def __init__(self, segments, root_video, clip_len=16, fps=3,
                 mode='clip', transform=None):
        self.data = segments.reset_index(drop=True)
        self.root = Path(root_video)
        self.clip_len = clip_len
        self.fps = fps
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = self.root / f"case_{row.VideoID}.mp4"
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found: {path}")

        # --- Convert annotation frame numbers (25 fps) → sampled frame indices ---
        seg_start = int(row.Start * self.fps / 25)
        seg_end   = int(row.End   * self.fps / 25)

        # --- Choose a start so that the whole clip fits inside the segment ---
        if seg_end - seg_start < self.clip_len:
            start = max(seg_start, seg_end - self.clip_len)  # short segment case
        else:
            # random start for better temporal variety
            start = np.random.randint(seg_start, seg_end - self.clip_len + 1)

        # --- Read frames ---
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for _ in range(self.clip_len):
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.resize(f, (224, 224))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)

        cap.release()

        # If clip shorter than expected, pad with last frame
        if len(frames) < self.clip_len:
            last = frames[-1] if frames else np.zeros((224, 224, 3), np.uint8)
            while len(frames) < self.clip_len:
                frames.append(last)

        # --- Convert to tensor ---
        x = np.stack(frames)                              # [T, H, W, C]
        x = torch.tensor(x).permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

        # --- Apply per-frame transform if provided ---
        if self.transform:
            x = torch.stack([self.transform(frame) for frame in x])

        # --- Rearrange for I3D input ---
        x = x.permute(1, 0, 2, 3).contiguous()            # [C, T, H, W]

        # --- Label (shift 1–10 → 0–9) ---
        y = int(row.PhaseID) - 1
        assert 0 <= y < 10, f"Invalid label {y} for video {row.VideoID}"

        return x, y

    
    def get_clips_for_video(self, video_id, clip_len=32, stride=16):
        """
        Extract overlapping clips from a given video.
        Each clip contains `clip_len` frames sampled at self.fps.
        Returns:
            clips  -> list of torch.Tensor of shape [clip_len, 3, H, W]
            starts -> list of starting frame indices
        """
        video_path = self.root / f"case_{video_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.arange(0, total_frames, int(self.fps_original / self.fps)) if hasattr(self, 'fps_original') else np.arange(0, total_frames)

        # Extract all frames at target fps
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        frames = np.stack(frames)

        # Slice into overlapping clips
        clips, starts = [], []
        for start in range(0, len(frames) - clip_len + 1, stride):
            clip = frames[start:start + clip_len]
            clip = torch.tensor(clip).permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
            if self.transform:
                clip = self.transform(clip)
            clips.append(clip)
            starts.append(start)
        return clips, starts
    
    def get_ground_truth_for_video(self, video_id):
        """
        Reconstruct the ground truth phase timeline (per frame)
        for the specified video from start/end frame annotations.

        Returns:
            gt_timeline -> np.array of phase IDs per frame index
        """
        video_segments = self.data[self.data["VideoID"] == video_id]
        if len(video_segments) == 0:
            raise ValueError(f"No annotations found for Video {video_id}")

        # Determine total frames from last segment
        total_frames = int(video_segments["End"].max())
        gt_timeline = np.zeros(total_frames, dtype=np.int32)

        for _, row in video_segments.iterrows():
            s, e, pid = int(row["Start"]), int(row["End"]), int(row["PhaseID"])
            gt_timeline[s:e] = pid
        return gt_timeline


def load_segments_from_csv(cfg_path="src/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))

    ann = pd.read_csv(cfg["paths"]["annotations"], sep=";")
    vids = pd.read_csv(cfg["paths"]["videos_csv"], sep=";")
    phases = pd.read_csv(cfg["paths"]["phases_csv"], sep=";")

    ann = ann.merge(phases, on="Phase", how="left")
    ann = ann.merge(vids[["VideoID", "Frames"]], on="VideoID")

    # reconstruct segments
    segments = []
    for vid, g in ann.groupby("VideoID"):
        g = g.sort_values("FrameNo").reset_index(drop=True)
        total = g["Frames"].iloc[0]
        for i in range(len(g)):
            start = g.loc[i, "FrameNo"]
            end = g.loc[i + 1, "FrameNo"] if i + 1 < len(g) else total
            segments.append([vid, start, end, g.loc[i, "Phase"], g.loc[i, "Meaning"]])
    segs = pd.DataFrame(segments, columns=["VideoID", "Start", "End", "PhaseID", "Phase"])
    return segs, Path(cfg["paths"]["video_root"]), cfg

