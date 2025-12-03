import pandas as pd
from pathlib import Path
from collections import Counter
from utils import plot_phase_distribution, plot_transition_matrix

def load_metadata(cfg):
    """Load CSVs defined in config.yaml."""
    ann = pd.read_csv(cfg['paths']['annotations'], sep=';')
    vids = pd.read_csv(cfg['paths']['videos'], sep=';')
    phases = pd.read_csv(cfg['paths']['phases'], sep=';')

    ann = ann.merge(phases, on='Phase', how='left')
    ann = ann.merge(vids[['VideoID','Frames']], on='VideoID')
    return ann, vids, phases

def reconstruct_segments(ann):
    """Build full segments (start→end) from annotations."""
    segments = []
    for vid, group in ann.groupby('VideoID'):
        group = group.sort_values('FrameNo').reset_index(drop=True)
        total = group['Frames'].iloc[0]
        for i in range(len(group)):
            start = group.loc[i, 'FrameNo']
            end = group.loc[i+1, 'FrameNo'] if i+1 < len(group) else total
            pid = group.loc[i, 'Phase']
            pname = group.loc[i, 'Meaning']
            segments.append([vid, start, end, pid, pname, end-start])
    segs = pd.DataFrame(segments, columns=['VideoID','Start','End','PhaseID','Phase','Length'])
    segs['Duration_s'] = segs['Length'] / 25
    return segs

def compute_transition_matrix(segments):
    """Compute frequency of phase-to-phase transitions."""
    transitions = Counter()
    for vid, group in segments.groupby('VideoID'):
        seq = list(group['Phase'])
        for i in range(len(seq)-1):
            transitions[(seq[i], seq[i+1])] += 1
    phases = sorted(segments['Phase'].unique())
    matrix = pd.DataFrame(0, index=phases, columns=phases)
    for (p1,p2),c in transitions.items():
        matrix.loc[p1,p2] = c
    return matrix

def run_eda(cfg):
    ann, vids, phases = load_metadata(cfg)
    segments = reconstruct_segments(ann)
    print(f"Total segments: {len(segments)} across {len(vids)} videos")

    # Stats
    stats = segments.groupby('Phase')['Duration_s'].agg(['mean','std','min','max','count'])
    stats.to_csv(Path(cfg['paths']['outputs']) / "duration_stats.csv")

    # Plots
    plot_phase_distribution(segments, Path(cfg['paths']['outputs']) / "class_balance.png")
    matrix = compute_transition_matrix(segments)
    plot_transition_matrix(matrix, Path(cfg['paths']['outputs']) / "transition_matrix.png")

    return segments, stats, matrix
