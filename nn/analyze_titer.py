#!/usr/bin/env python3
"""
Deep-dive analysis of Titer prediction problems.
Compare Titer vs Glucose patterns to understand why Titer is harder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from load_real_data import load_experimental_data
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def analyze_titer():
    """Comprehensive Titer analysis."""
    
    # Load data
    possible_paths = [
        Path("data_2.csv"),
        Path("/Users/nevecallaway/COSMIC-dFBA-nn/COSMIC-dFBA-nn/nn/data_2.csv"),
    ]
    
    data_file = None
    for p in possible_paths:
        if p.exists():
            data_file = str(p)
            break
    
    if data_file is None:
        print("Error: data_2.csv not found")
        sys.exit(1)
    
    df = pd.read_csv(data_file)
    
    # Get unique reactors
    reactors = sorted(df['Vessel'].unique())
    print(f"\n{'='*80}")
    print("TITER DEEP-DIVE ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Reactors found: {reactors}\n")
    
    # Component columns (in normalized space)
    component_cols = ['Cell Density', 'Glucose', 'Lactate', 'Titer']
    
    # ============================================================================
    # 1. TITER VS GLUCOSE: RAW STATISTICS
    # ============================================================================
    print(f"{'='*80}")
    print("1. RAW STATISTICS: TITER vs GLUCOSE")
    print(f"{'='*80}\n")
    
    for comp in ['Titer', 'Glucose']:
        data = df[comp]
        print(f"{comp}:")
        print(f"  Mean:        {data.mean():.4f}")
        print(f"  Std:         {data.std():.4f}")
        print(f"  Min:         {data.min():.4f}")
        print(f"  Max:         {data.max():.4f}")
        print(f"  Median:      {data.median():.4f}")
        print(f"  25th pct:    {data.quantile(0.25):.4f}")
        print(f"  75th pct:    {data.quantile(0.75):.4f}")
        
        # Count values in different ranges
        print(f"  Value distribution:")
        print(f"    < 0.2:     {(data < 0.2).sum():4d} points ({100*(data < 0.2).sum()/len(data):.1f}%)")
        print(f"    0.2-0.4:   {((data >= 0.2) & (data < 0.4)).sum():4d} points ({100*((data >= 0.2) & (data < 0.4)).sum()/len(data):.1f}%)")
        print(f"    0.4-0.6:   {((data >= 0.4) & (data < 0.6)).sum():4d} points ({100*((data >= 0.4) & (data < 0.6)).sum()/len(data):.1f}%)")
        print(f"    0.6-0.8:   {((data >= 0.6) & (data < 0.8)).sum():4d} points ({100*((data >= 0.6) & (data < 0.8)).sum()/len(data):.1f}%)")
        print(f"    >= 0.8:    {(data >= 0.8).sum():4d} points ({100*(data >= 0.8).sum()/len(data):.1f}%)")
        print()
    
    # ============================================================================
    # 2. TITER PATTERNS BY REACTOR
    # ============================================================================
    print(f"{'='*80}")
    print("2. TITER PATTERNS BY REACTOR")
    print(f"{'='*80}\n")
    
    reactor_stats = []
    for reactor in reactors:
        reactor_data = df[df['Vessel'] == reactor]
        titer = reactor_data['Titer']
        glucose = reactor_data['Glucose']
        
        stats = {
            'Reactor': reactor,
            'Titer_Mean': titer.mean(),
            'Titer_Std': titer.std(),
            'Titer_Range': titer.max() - titer.min(),
            'Titer_Low': (titer < 0.2).sum() / len(titer),
            'Glucose_Mean': glucose.mean(),
            'Glucose_Std': glucose.std(),
            'Titer_Glucose_Corr': np.corrcoef(titer, glucose)[0, 1] if len(titer) > 1 else 0,
        }
        reactor_stats.append(stats)
    
    reactor_stats_df = pd.DataFrame(reactor_stats)
    
    print("Reactor-by-Reactor Analysis:")
    print(reactor_stats_df.to_string(index=False))
    print()
    
    # Identify easy vs hard reactors
    print("Easy reactors (high variance, good spread):")
    easy = reactor_stats_df[reactor_stats_df['Titer_Range'] > reactor_stats_df['Titer_Range'].quantile(0.75)]
    for _, row in easy.iterrows():
        print(f"  {row['Reactor']}: range={row['Titer_Range']:.3f}, mean={row['Titer_Mean']:.3f}, std={row['Titer_Std']:.3f}")
    print()
    
    print("Hard reactors (low variance, clustered):")
    hard = reactor_stats_df[reactor_stats_df['Titer_Range'] <= reactor_stats_df['Titer_Range'].quantile(0.25)]
    for _, row in hard.iterrows():
        print(f"  {row['Reactor']}: range={row['Titer_Range']:.3f}, mean={row['Titer_Mean']:.3f}, std={row['Titer_Std']:.3f}")
    print()
    
    # ============================================================================
    # 3. TITER DYNAMICS: HOW DOES TITER CHANGE OVER TIME?
    # ============================================================================
    print(f"{'='*80}")
    print("3. TITER DYNAMICS (Time-series patterns)")
    print(f"{'='*80}\n")
    
    time_stats = []
    times = sorted(df['Time'].unique())
    
    for t in times:
        time_data = df[df['Time'] == t]
        titer_t = time_data['Titer']
        glucose_t = time_data['Glucose']
        phase_t = time_data['Production phase fraction']
        
        time_stats.append({
            'Time': t,
            'Titer_Mean': titer_t.mean(),
            'Titer_Std': titer_t.std(),
            'Titer_Range': titer_t.max() - titer_t.min(),
            'Glucose_Mean': glucose_t.mean(),
            'Phase_Mean': phase_t.mean(),
            'N': len(time_data),
        })
    
    time_stats_df = pd.DataFrame(time_stats)
    
    print("Time-point Statistics:")
    print(time_stats_df.to_string(index=False))
    print()
    
    # ============================================================================
    # 4. PHASE-SPECIFIC TITER ANALYSIS
    # ============================================================================
    print(f"{'='*80}")
    print("4. PHASE-SPECIFIC ANALYSIS")
    print(f"{'='*80}\n")
    
    # Bin by phase
    df['Phase_Bin'] = pd.cut(df['Production phase fraction'], 
                              bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
                              labels=['Early(0-0.1)', 'Growth(0.1-0.3)', 'Mid(0.3-0.5)', 'Late(0.5-0.7)', 'End(0.7-1.0)'])
    
    phase_analysis = []
    for phase_bin in ['Early(0-0.1)', 'Growth(0.1-0.3)', 'Mid(0.3-0.5)', 'Late(0.5-0.7)', 'End(0.7-1.0)']:
        phase_data = df[df['Phase_Bin'] == phase_bin]
        if len(phase_data) > 0:
            titer_p = phase_data['Titer']
            glucose_p = phase_data['Glucose']
            phase_analysis.append({
                'Phase': phase_bin,
                'N': len(phase_data),
                'Titer_Mean': titer_p.mean(),
                'Titer_Std': titer_p.std(),
                'Titer_CV': titer_p.std() / titer_p.mean() if titer_p.mean() > 0 else np.inf,
                'Glucose_Mean': glucose_p.mean(),
                'Titer_Clustering': (titer_p < 0.15).sum() / len(titer_p),  # % low values
            })
    
    phase_analysis_df = pd.DataFrame(phase_analysis)
    print("Phase-Binned Analysis:")
    print(phase_analysis_df.to_string(index=False))
    print()
    
    # ============================================================================
    # 5. CORRELATION STRUCTURE
    # ============================================================================
    print(f"{'='*80}")
    print("5. CORRELATION ANALYSIS")
    print(f"{'='*80}\n")
    
    numeric_cols = ['Cell Density', 'Glucose', 'Lactate', 'Titer', 'Production phase fraction']
    corr_matrix = df[numeric_cols].corr()
    
    print("Full correlation matrix:")
    print(corr_matrix)
    print()
    
    print("Titer correlations with other variables:")
    titer_corr = corr_matrix['Titer'].sort_values(ascending=False)
    print(titer_corr)
    print()
    
    print("Glucose correlations with other variables:")
    glucose_corr = corr_matrix['Glucose'].sort_values(ascending=False)
    print(glucose_corr)
    print()
    
    # ============================================================================
    # 6. CLUSTERING ANALYSIS: TITER HISTOGRAM
    # ============================================================================
    print(f"{'='*80}")
    print("6. VALUE CLUSTERING ANALYSIS")
    print(f"{'='*80}\n")
    
    titer_vals = df['Titer'].values
    glucose_vals = df['Glucose'].values
    
    print("Titer value clustering:")
    print(f"  < 0.05:   {(titer_vals < 0.05).sum():4d} points (highly clustered)")
    print(f"  0.05-0.15: {((titer_vals >= 0.05) & (titer_vals < 0.15)).sum():4d} points")
    print(f"  0.15-0.25: {((titer_vals >= 0.15) & (titer_vals < 0.25)).sum():4d} points")
    print(f"  0.25-0.50: {((titer_vals >= 0.25) & (titer_vals < 0.50)).sum():4d} points")
    print(f"  > 0.50:    {(titer_vals >= 0.50).sum():4d} points (rare)")
    print()
    
    print("Glucose value distribution (for comparison):")
    print(f"  < 0.05:   {(glucose_vals < 0.05).sum():4d} points")
    print(f"  0.05-0.25: {((glucose_vals >= 0.05) & (glucose_vals < 0.25)).sum():4d} points")
    print(f"  0.25-0.50: {((glucose_vals >= 0.25) & (glucose_vals < 0.50)).sum():4d} points")
    print(f"  0.50-0.75: {((glucose_vals >= 0.50) & (glucose_vals < 0.75)).sum():4d} points")
    print(f"  > 0.75:    {(glucose_vals >= 0.75).sum():4d} points")
    print()
    
    # ============================================================================
    # 7. KEY FINDINGS
    # ============================================================================
    print(f"{'='*80}")
    print("KEY FINDINGS - WHY TITER IS HARD")
    print(f"{'='*80}\n")
    
    findings = []
    
    # Finding 1: Clustering
    pct_low = (titer_vals < 0.2).sum() / len(titer_vals) * 100
    findings.append(f"1. EXTREME CLUSTERING: {pct_low:.1f}% of Titer values < 0.2 (bunched in low range)")
    
    # Finding 2: Low variance
    findings.append(f"2. LOW VARIANCE: Titer std={df['Titer'].std():.4f} vs Glucose std={df['Glucose'].std():.4f}")
    
    # Finding 3: Early phase
    early_phase_data = df[df['Production phase fraction'] < 0.1]
    if len(early_phase_data) > 0:
        early_titer_mean = early_phase_data['Titer'].mean()
        early_titer_std = early_phase_data['Titer'].std()
        findings.append(f"3. EARLY PHASE CHAOS: Early phase Titer mean={early_titer_mean:.4f}, std={early_titer_std:.4f} (highly variable, hard to predict)")
    
    # Finding 4: Weak correlations
    titer_cell_corr = np.corrcoef(df['Titer'], df['Cell Density'])[0, 1]
    titer_glucose_corr = np.corrcoef(df['Titer'], df['Glucose'])[0, 1]
    glucose_cell_corr = np.corrcoef(df['Glucose'], df['Cell Density'])[0, 1]
    findings.append(f"4. WEAK INPUT SIGNALS: Titer-CellDensity corr={titer_cell_corr:.3f}, Titer-Glucose corr={titer_glucose_corr:.3f}")
    findings.append(f"   (vs Glucose-CellDensity corr={glucose_cell_corr:.3f})")
    
    # Finding 5: Reactor variance
    easy_titer_range = easy['Titer_Range'].mean() if len(easy) > 0 else 0
    hard_titer_range = hard['Titer_Range'].mean() if len(hard) > 0 else 0
    findings.append(f"5. REACTOR DEPENDENCY: Easy reactors avg range={easy_titer_range:.3f}, Hard reactors avg range={hard_titer_range:.3f}")
    
    # Finding 6: Saturation
    pct_high = (titer_vals > 0.9).sum() / len(titer_vals) * 100
    findings.append(f"6. SATURATION: {pct_high:.1f}% of values saturate at high values")
    
    for finding in findings:
        print(finding)
    
    print(f"\n{'='*80}\n")
    
    return df, reactor_stats_df, time_stats_df


if __name__ == "__main__":
    analyze_titer()
