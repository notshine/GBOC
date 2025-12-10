import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.patches import Rectangle

class AnomalyScoreVisualizer:
    def __init__(self, data, anomaly_scores, labels, save_path=None):
        self.data = data
        self.anomaly_scores = anomaly_scores
        self.labels = labels
        self.save_path = save_path
        
        # Set font for Chinese characters (if needed)
        plt.rcParams['font.sans-serif'] = ['Calibri', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_overview(self, figsize=(18, 9)):
        """Plot complete overview"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Calculate anomaly regions
        _n = len(self.labels)
        x = np.arange(_n)
        
        # Upper plot: original data + anomaly regions
        if len(self.data.shape) > 1:
            data_to_plot = self.data[:, 0]  # Take first column
        else:
            data_to_plot = self.data
            
        min_val = np.min(data_to_plot)
        max_val = np.max(data_to_plot)
        y1 = np.full(_n, min_val)
        y2 = np.array([max_val * self.labels[i] for i in range(_n)])
        
        ax1.fill_between(x, y1, y2, where=(self.labels == 1), alpha=0.7, color='red', label='Ground Truth Anomaly')
        ax1.plot(data_to_plot, color='blue', label='Time Series Data')
        # ax1.set_title('Original Data with Anomaly Regions', fontsize=14)
        # ax1.set_ylabel('Value', fontsize=12)
        ax1.legend(loc="upper left", fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        # ax1.grid(True, alpha=0.3)
        
        # Lower plot: anomaly scores
        ax2.plot(self.anomaly_scores, color='red', label='Anomaly Scores')
        # ax2.set_title('Anomaly Scores (Click and drag to select region for zoom)', fontsize=14)
        # ax2.set_ylabel('Anomaly Score', fontsize=12)
        # ax2.set_xlabel('Time Index', fontsize=12)
        ax2.legend(loc="upper left", fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        # ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f"{self.save_path}_overview.png", dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def plot_local_detail(self, start_idx, end_idx, figsize=(15, 10)):
        """Plot local detailed view"""
        # Ensure indices are within valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(self.data), end_idx)
        
        if start_idx >= end_idx:
            print("Invalid index range")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Local data
        x_local = np.arange(start_idx, end_idx)
        
        if len(self.data.shape) > 1:
            data_local = self.data[start_idx:end_idx, 0]
        else:
            data_local = self.data[start_idx:end_idx]
            
        scores_local = self.anomaly_scores[start_idx:end_idx]
        labels_local = self.labels[start_idx:end_idx]
        
        # Upper plot: local original data
        min_val = np.min(data_local)
        max_val = np.max(data_local)
        y1 = np.full(len(x_local), min_val)
        y2 = np.array([max_val * labels_local[i] for i in range(len(labels_local))])
        
        ax1.fill_between(x_local, y1, y2, where=(labels_local == 1), 
                        alpha=0.3, color='red', label='Anomaly Area')
        ax1.plot(x_local, data_local, color='blue', linewidth=1.2, 
                marker='o', markersize=2, label='Original Data')
        ax1.set_title(f'Original Data Detail View (Index {start_idx}-{end_idx})', fontsize=14)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Lower plot: local anomaly scores
        ax2.plot(x_local, scores_local, color='red', linewidth=1.5, 
                marker='o', markersize=2, label='Anomaly Scores')
        
        # Mark anomaly points
        anomaly_indices = x_local[labels_local == 1]
        if len(anomaly_indices) > 0:
            anomaly_scores_at_anomalies = scores_local[labels_local == 1]
            # ax2.scatter(anomaly_indices, anomaly_scores_at_anomalies, 
            #            color='darkred', s=30, zorder=5, label='True Anomaly Points')
        
        ax2.set_title(f'Anomaly Scores Detail View (Index {start_idx}-{end_idx})', fontsize=14)
        ax2.set_ylabel('Anomaly Score', fontsize=12)
        ax2.set_xlabel('Time Index', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f"{self.save_path}_detail_{start_idx}_{end_idx}.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def plot_downsampled(self, step=50, figsize=(15, 10)):
        """Plot downsampled data"""
        indices = np.arange(0, len(self.data), step)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Downsampled data
        if len(self.data.shape) > 1:
            data_sampled = self.data[::step, 0]
        else:
            data_sampled = self.data[::step]
            
        scores_sampled = self.anomaly_scores[::step]
        labels_sampled = self.labels[::step]
        
        # Upper plot
        min_val = np.min(data_sampled)
        max_val = np.max(data_sampled)
        y1 = np.full(len(indices), min_val)
        y2 = np.array([max_val * labels_sampled[i] for i in range(len(labels_sampled))])
        
        ax1.fill_between(indices, y1, y2, where=(labels_sampled == 1), 
                        alpha=0.3, color='red', label='Anomaly Area')
        ax1.plot(indices, data_sampled, color='blue', linewidth=1.2, 
                marker='o', markersize=3, label='Original Data')
        ax1.set_title(f'Original Data (Sampled every {step} points)', fontsize=14)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Lower plot
        ax2.plot(indices, scores_sampled, color='red', linewidth=1.5, 
                marker='o', markersize=3, label='Anomaly Scores')
        ax2.set_title(f'Anomaly Scores (Sampled every {step} points)', fontsize=14)
        ax2.set_ylabel('Anomaly Score', fontsize=12)
        ax2.set_xlabel('Time Index', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f"{self.save_path}_downsampled_{step}.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)

def interactive_plot(data, anomaly_scores, labels, save_path=None):
    """Interactive plotting function"""
    visualizer = AnomalyScoreVisualizer(data, anomaly_scores, labels, save_path)
    
    # Create overview plot
    fig, (ax1, ax2) = visualizer.plot_overview()
    
    def onselect(eclick, erelease):
        """Selection callback function"""
        if eclick.inaxes != ax2:
            return
            
        x1, x2 = int(eclick.xdata), int(erelease.xdata)
        if x1 > x2:
            x1, x2 = x2, x1
            
        print(f"Selected range: {x1} - {x2}")
        
        # Create local detail plot
        visualizer.plot_local_detail(x1, x2)
        plt.show()
    
    # Add rectangle selector
    selector = RectangleSelector(ax2, onselect, useblit=True, 
                               button=[1], minspanx=5, minspany=0,
                               spancoords='pixels', interactive=True)
    
    plt.show()
    
    return visualizer

# Usage example function
def plot_anomaly_scores_from_files(score_file, data_file=None, label_file=None, 
                                 start_idx=None, end_idx=None, step=None):
    """Load data from files and plot"""
    # Load anomaly scores
    if score_file.endswith('.npy'):
        scores = np.load(score_file)
    else:
        scores = np.loadtxt(score_file)
    
    # If no data and label files provided, create simulated data
    if data_file is None:
        data = np.random.randn(len(scores))
        print("Warning: No data file provided, using random data")
    else:
        data = np.load(data_file) if data_file.endswith('.npy') else np.loadtxt(data_file)
    
    if label_file is None:
        labels = np.zeros(len(scores))
        print("Warning: No label file provided, using zero labels")
    else:
        labels = np.load(label_file) if label_file.endswith('.npy') else np.loadtxt(label_file)
    
    # Create visualizer
    save_path = score_file.replace('.npy', '').replace('.txt', '')
    visualizer = AnomalyScoreVisualizer(data, scores, labels, save_path)
    
    if start_idx is not None and end_idx is not None:
        # Plot local detail
        visualizer.plot_local_detail(start_idx, end_idx)
    elif step is not None:
        # Plot downsampled
        visualizer.plot_downsampled(step)
    else:
        # Interactive plotting
        interactive_plot(data, scores, labels, save_path)

if __name__ == "__main__":
    # Usage example
    print("Anomaly Score Visualization Tool")
    print("1. Interactive plotting (recommended)")
    print("2. Local zoom")
    print("3. Downsampled display")
    
    # Example: if you have specific file paths
    # plot_anomaly_scores_from_files('path/to/your/scores.npy')
    
    # Or use data directly
    # visualizer = AnomalyScoreVisualizer(your_data, your_scores, your_labels)
    # interactive_plot(your_data, your_scores, your_labels)