import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from preprocess import get_cifar10_data, prepare_batch_for_gpu
from train import train_classifier, train_diffusion_model, PurifyCModel
from evaluate import (generate_adversarial_examples, evaluate_purification_performance, 
                     analyze_lambda_strategies, temporal_confidence_analysis, calculate_metrics)

def setup_device():
    """Setup device for Tesla T4 compatibility."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        torch.cuda.set_per_process_memory_fraction(0.8)
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def create_high_quality_plots(results, lambda_histories, confidence_histories, 
                            confidence_evolution, lambda_evolution, save_dir):
    """Generate high-quality PDF plots for academic papers."""
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'figure.figsize': (10, 8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    schedules = list(results.keys())
    clean_acc = [results[s]['clean_accuracy'] for s in schedules]
    adv_acc = [results[s]['adversarial_accuracy'] for s in schedules]
    purified_acc = [results[s]['purified_accuracy'] for s in schedules]
    
    x = np.arange(len(schedules))
    width = 0.25
    
    ax1.bar(x - width, clean_acc, width, label='Clean Accuracy', alpha=0.8)
    ax1.bar(x, adv_acc, width, label='Adversarial Accuracy', alpha=0.8)
    ax1.bar(x + width, purified_acc, width, label='Purified Accuracy', alpha=0.8)
    
    ax1.set_xlabel('Lambda Scheduling Strategy')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Purify-C Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(schedules)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    conf_imp = [results[s]['confidence_improvement'] for s in schedules]
    ax2.bar(schedules, conf_imp, alpha=0.8, color='orange')
    ax2.set_xlabel('Lambda Scheduling Strategy')
    ax2.set_ylabel('Confidence Improvement')
    ax2.set_title('Confidence Improvement by Strategy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'experiment1_performance_comparison.pdf'), 
                format='pdf', bbox_inches='tight')
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    steps = range(len(lambda_histories['exponential']))
    
    for strategy in ['exponential', 'polynomial', 'sigmoid']:
        lambda_mean = np.mean(lambda_histories[strategy], axis=1)
        lambda_std = np.std(lambda_histories[strategy], axis=1)
        
        ax1.plot(steps, lambda_mean, label=f'{strategy.capitalize()}', linewidth=2)
        ax1.fill_between(steps, lambda_mean - lambda_std, lambda_mean + lambda_std, alpha=0.2)
    
    ax1.set_xlabel('Diffusion Step')
    ax1.set_ylabel('Lambda Value')
    ax1.set_title('Lambda Scheduling Strategies During Purification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for strategy in ['exponential', 'polynomial', 'sigmoid']:
        conf_mean = np.mean(confidence_histories[strategy], axis=1)
        conf_std = np.std(confidence_histories[strategy], axis=1)
        
        ax2.plot(steps, conf_mean, label=f'{strategy.capitalize()}', linewidth=2)
        ax2.fill_between(steps, conf_mean - conf_std, conf_mean + conf_std, alpha=0.2)
    
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('Classifier Confidence')
    ax2.set_title('Confidence Evolution During Purification')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'experiment2_lambda_strategies.pdf'), 
                format='pdf', bbox_inches='tight')
    plt.close()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    steps = range(len(confidence_evolution))
    
    conf_mean = np.mean(confidence_evolution, axis=1)
    conf_std = np.std(confidence_evolution, axis=1)
    ax1.plot(steps, conf_mean, 'b-', linewidth=2, label='Mean Confidence')
    ax1.fill_between(steps, conf_mean - conf_std, conf_mean + conf_std, alpha=0.3)
    ax1.set_ylabel('Classifier Confidence')
    ax1.set_title('Temporal Analysis: Confidence Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    lambda_mean = np.mean(lambda_evolution, axis=1)
    lambda_std = np.std(lambda_evolution, axis=1)
    ax2.plot(steps, lambda_mean, 'r-', linewidth=2, label='Mean Lambda')
    ax2.fill_between(steps, lambda_mean - lambda_std, lambda_mean + lambda_std, alpha=0.3)
    ax2.set_ylabel('Lambda Value')
    ax2.set_title('Lambda Adaptation During Purification')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    correlation = []
    for i in range(len(confidence_evolution)):
        if len(confidence_evolution[i]) > 1 and len(lambda_evolution[i]) > 1:
            corr = np.corrcoef(confidence_evolution[i], lambda_evolution[i])[0, 1]
            correlation.append(corr if not np.isnan(corr) else 0)
        else:
            correlation.append(0)
    
    ax3.plot(steps, correlation, 'g-', linewidth=2, label='Confidence-Lambda Correlation')
    ax3.set_xlabel('Diffusion Step')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.set_title('Confidence-Lambda Correlation Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'experiment3_temporal_analysis.pdf'), 
                format='pdf', bbox_inches='tight')
    plt.close()
    
    print("All high-quality PDF plots saved successfully!")

def run_experiment_1(purify_model, classifier, testloader, device):
    """Experiment 1: Comparative purification performance."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Comparative Purification Performance")
    print("="*60)
    
    print("Generating adversarial examples...")
    adv_examples, clean_examples, labels = generate_adversarial_examples(
        classifier, testloader, device, attack_type='fgsm', epsilon=0.03
    )
    
    print("Evaluating purification performance...")
    results = evaluate_purification_performance(
        purify_model, classifier, adv_examples, clean_examples, labels, device
    )
    
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Strategy':<12} {'Clean Acc':<10} {'Adv Acc':<10} {'Purified Acc':<12} {'Conf Imp':<10}")
    print("-" * 80)
    
    for strategy, result in results.items():
        print(f"{strategy:<12} {result['clean_accuracy']:<10.3f} {result['adversarial_accuracy']:<10.3f} "
              f"{result['purified_accuracy']:<12.3f} {result['confidence_improvement']:<10.3f}")
    
    return results, adv_examples

def run_experiment_2(purify_model, adv_examples, device):
    """Experiment 2: Ablation study on lambda strategies."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Lambda Strategy Ablation Study")
    print("="*60)
    
    lambda_histories, confidence_histories = analyze_lambda_strategies(
        purify_model, adv_examples, device
    )
    
    print("\nLambda Strategy Analysis:")
    print("-" * 50)
    
    for strategy in ['exponential', 'polynomial', 'sigmoid']:
        lambda_mean = np.mean(lambda_histories[strategy])
        lambda_std = np.std(lambda_histories[strategy])
        conf_mean = np.mean(confidence_histories[strategy])
        conf_std = np.std(confidence_histories[strategy])
        
        print(f"{strategy.capitalize()}:")
        print(f"  Lambda: {lambda_mean:.4f} ± {lambda_std:.4f}")
        print(f"  Confidence: {conf_mean:.4f} ± {conf_std:.4f}")
    
    return lambda_histories, confidence_histories

def run_experiment_3(purify_model, adv_examples, device):
    """Experiment 3: Temporal analysis of confidence evolution."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Temporal Confidence Analysis")
    print("="*60)
    
    confidence_evolution, lambda_evolution = temporal_confidence_analysis(
        purify_model, adv_examples, device
    )
    
    print("\nTemporal Analysis Results:")
    print("-" * 40)
    print(f"Initial confidence: {np.mean(confidence_evolution[0]):.4f}")
    print(f"Final confidence: {np.mean(confidence_evolution[-1]):.4f}")
    print(f"Confidence improvement: {np.mean(confidence_evolution[-1]) - np.mean(confidence_evolution[0]):.4f}")
    print(f"Average lambda: {np.mean(lambda_evolution):.4f}")
    
    return confidence_evolution, lambda_evolution

def update_research_status():
    """Update research_history.json with completion status."""
    history_path = '.research/research_history.json'
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        history['status_enum'] = 'stopped'
        history['completion_time'] = datetime.now().isoformat()
        history['experiment_status'] = 'completed'
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nResearch status updated: status_enum set to 'stopped'")
        
    except Exception as e:
        print(f"Warning: Could not update research status: {e}")

def main():
    """Main experimental pipeline for Purify-C validation."""
    print("="*80)
    print("PURIFY-C: CONFIDENCE-GUIDED DIFFUSION PURIFICATION")
    print("Experimental Validation Pipeline")
    print("="*80)
    
    device = setup_device()
    torch.manual_seed(42)
    np.random.seed(42)
    
    save_dir = '.research/iteration1/images'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nLoading CIFAR-10 dataset...")
    trainloader, testloader = get_cifar10_data(batch_size=32, subset_size=1000)
    
    print("\nTraining classifier...")
    classifier = train_classifier(trainloader, device, epochs=3)
    
    print("\nTraining diffusion model...")
    diffusion_model = train_diffusion_model(trainloader, device, epochs=2)
    
    print("\nInitializing Purify-C model...")
    purify_model = PurifyCModel(classifier, diffusion_model, device)
    
    results, adv_examples = run_experiment_1(purify_model, classifier, testloader, device)
    lambda_histories, confidence_histories = run_experiment_2(purify_model, adv_examples, device)
    confidence_evolution, lambda_evolution = run_experiment_3(purify_model, adv_examples, device)
    
    print("\nGenerating high-quality PDF plots...")
    create_high_quality_plots(
        results, lambda_histories, confidence_histories,
        confidence_evolution, lambda_evolution, save_dir
    )
    
    metrics = calculate_metrics(results)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print("\nOverall Performance Metrics:")
    print("-" * 50)
    for strategy, metric in metrics.items():
        print(f"{strategy.capitalize()}:")
        print(f"  Robustness Improvement: {metric['robustness_improvement']:.4f}")
        print(f"  Clean Degradation: {metric['clean_degradation']:.4f}")
        print(f"  Overall Score: {metric['overall_score']:.4f}")
    
    best_strategy = max(metrics.keys(), key=lambda k: metrics[k]['overall_score'])
    print(f"\nBest performing strategy: {best_strategy.capitalize()}")
    print(f"Overall score: {metrics[best_strategy]['overall_score']:.4f}")
    
    update_research_status()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("All results saved to .research/iteration1/images/")
    print("Status updated: status_enum = 'stopped'")
    print("="*80)

if __name__ == "__main__":
    main()
