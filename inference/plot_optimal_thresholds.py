import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jsonlines
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

def load_prediction_file(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append({
                'label': obj['label'],
                'probability_vulnerable': obj['probability_vulnerable']
            })
    return data

def calculate_metrics_for_threshold(data, threshold):
    y_true = [item['label'] for item in data]
    y_pred = [1 if item['probability_vulnerable'] >= threshold else 0 for item in data]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def find_optimal_threshold(data, metric='accuracy', num_points=1000):
    thresholds = np.linspace(0, 1, num_points)
    metrics = []
    
    for threshold in thresholds:
        result = calculate_metrics_for_threshold(data, threshold)
        metrics.append(result)
    
    metrics_df = pd.DataFrame(metrics)
    optimal_idx = metrics_df[metric].idxmax()
    optimal_threshold = metrics_df.iloc[optimal_idx]['threshold']
    optimal_metrics = metrics_df.iloc[optimal_idx].to_dict()
    
    return optimal_metrics, metrics_df

def plot_threshold_metrics(metrics_df, dataset_name, save_path=None):
    plt.figure(figsize=(10, 6))
    
    plt.plot(metrics_df['threshold'], metrics_df['accuracy'], label='Accuracy', linewidth=2)
    plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', linewidth=2)
    plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', linewidth=2)
    plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1', linewidth=2)
    
    optimal_idx = metrics_df['accuracy'].idxmax()
    optimal_threshold = metrics_df.iloc[optimal_idx]['threshold']
    optimal_accuracy = metrics_df.iloc[optimal_idx]['accuracy']
    
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal Threshold: {optimal_threshold:.3f}')
    plt.axhline(y=optimal_accuracy, color='g', linestyle='--', 
                label=f'Max Accuracy: {optimal_accuracy:.3f}')
    
    plt.title(f'Threshold Analysis for {dataset_name}')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_all_datasets_comparison(datasets_metrics, save_path=None):
    plt.figure(figsize=(12, 8))
    
    for dataset_name, metrics_df in datasets_metrics.items():
        plt.plot(metrics_df['threshold'], metrics_df['accuracy'], label=f'{dataset_name}', linewidth=2)
        
        optimal_idx = metrics_df['accuracy'].idxmax()
        optimal_threshold = metrics_df.iloc[optimal_idx]['threshold']
        optimal_accuracy = metrics_df.iloc[optimal_idx]['accuracy']
        
        plt.scatter(optimal_threshold, optimal_accuracy, marker='o', s=80)
        plt.annotate(f'{optimal_threshold:.2f}', 
                    (optimal_threshold, optimal_accuracy),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.title('Accuracy vs Threshold Across Datasets')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_optimal_thresholds(datasets_optimal, output_file):
    results = []
    
    for dataset_name, optimal in datasets_optimal.items():
        results.append({
            'dataset': dataset_name,
            'optimal_threshold': optimal['threshold'],
            'accuracy': optimal['accuracy'],
            'precision': optimal['precision'],
            'recall': optimal['recall'],
            'f1': optimal['f1']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Optimal thresholds saved to {output_file}")
    
    # Print as a table for quick viewing
    print("\nOptimal Thresholds Summary:")
    print(df.to_string())

def main():
    parser = argparse.ArgumentParser(description='Analyze prediction files and find optimal thresholds')
    parser.add_argument('--predictions-dir', default="./results", help='Directory containing prediction JSONL files')
    parser.add_argument('--output-dir', default='./threshold_analysis', help='Directory to save outputs')
    parser.add_argument('--metric', default='accuracy', choices=['accuracy', 'f1', 'precision', 'recall'], 
                        help='Metric to optimize for finding optimal threshold')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    prediction_files = glob.glob(os.path.join(args.predictions_dir, '*_predictions.jsonl'))
    if not prediction_files:
        print(f"No prediction files found in {args.predictions_dir}")
        return
    
    datasets_metrics = {}
    datasets_optimal = {}
    
    for pred_file in prediction_files:
        dataset_name = os.path.basename(pred_file).replace('_predictions.jsonl', '')
        print(f"Processing {dataset_name}...")
        
        data = load_prediction_file(pred_file)
        optimal_metrics, metrics_df = find_optimal_threshold(data, metric=args.metric)
        
        datasets_metrics[dataset_name] = metrics_df
        datasets_optimal[dataset_name] = optimal_metrics
        
        save_path = os.path.join(args.output_dir, f'{dataset_name}_threshold_analysis.png')
        plot_threshold_metrics(metrics_df, dataset_name, save_path)
        
        print(f"  Optimal threshold: {optimal_metrics['threshold']:.4f} with {args.metric}: {optimal_metrics[args.metric]:.4f}")
    
    comparison_path = os.path.join(args.output_dir, 'datasets_comparison_threshold_analysis.png')
    plot_all_datasets_comparison(datasets_metrics, comparison_path)
    
    optimal_path = os.path.join(args.output_dir, 'optimal_thresholds.csv')
    save_optimal_thresholds(datasets_optimal, optimal_path)

if __name__ == "__main__":
    main()
