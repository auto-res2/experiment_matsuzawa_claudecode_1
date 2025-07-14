import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def fgsm_attack(model, data, target, epsilon=0.03):
    """Fast Gradient Sign Method attack."""
    data.requires_grad = True
    output = model(data)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

def pgd_attack(model, data, target, epsilon=0.03, alpha=0.01, num_iter=10):
    """Projected Gradient Descent attack."""
    original_data = data.clone()
    
    for i in range(num_iter):
        data.requires_grad = True
        output = model(data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        
        data = data + alpha * data_grad.sign()
        eta = torch.clamp(data - original_data, min=-epsilon, max=epsilon)
        data = torch.clamp(original_data + eta, min=0, max=1).detach()
    
    return data

def generate_adversarial_examples(classifier, dataloader, device, attack_type='fgsm', epsilon=0.03):
    """Generate adversarial examples using FGSM or PGD."""
    classifier.eval()
    
    adv_examples = []
    clean_examples = []
    labels = []
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=f"Generating {attack_type.upper()} attacks")):
        if batch_idx >= 10:  # Limit for memory efficiency
            break
            
        data, target = data.to(device), target.to(device)
        
        if attack_type == 'fgsm':
            adv_data = fgsm_attack(classifier, data, target, epsilon)
        elif attack_type == 'pgd':
            adv_data = pgd_attack(classifier, data, target, epsilon)
        else:
            raise ValueError("Attack type must be 'fgsm' or 'pgd'")
        
        adv_examples.append(adv_data.cpu())
        clean_examples.append(data.cpu())
        labels.append(target.cpu())
    
    return torch.cat(adv_examples), torch.cat(clean_examples), torch.cat(labels)

def evaluate_purification_performance(purify_model, classifier, adv_examples, clean_examples, labels, device):
    """Evaluate purification performance comparing different lambda schedules."""
    classifier.eval()
    purify_model.diffusion_model.eval()
    
    schedules = ['fixed', 'exponential', 'polynomial', 'sigmoid']
    results = {}
    
    for schedule in schedules:
        print(f"Evaluating {schedule} lambda schedule...")
        
        correct_clean = 0
        correct_adv = 0
        correct_purified = 0
        total = 0
        
        confidence_improvements = []
        
        batch_size = 8  # Small batch for memory efficiency
        num_batches = min(len(adv_examples) // batch_size, 20)
        
        for i in tqdm(range(num_batches), desc=f"{schedule} evaluation"):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_adv = adv_examples[start_idx:end_idx].to(device)
            batch_clean = clean_examples[start_idx:end_idx].to(device)
            batch_labels = labels[start_idx:end_idx].to(device)
            
            with torch.no_grad():
                clean_pred = classifier(batch_clean).argmax(dim=1)
                correct_clean += (clean_pred == batch_labels).sum().item()
                
                adv_pred = classifier(batch_adv).argmax(dim=1)
                correct_adv += (adv_pred == batch_labels).sum().item()
                
                conf_before = classifier.get_confidence(batch_adv)
                
                if schedule == 'fixed':
                    purified, _, _ = purify_model.purify(batch_adv, num_steps=20, lambda_schedule='fixed')
                else:
                    purified, _, _ = purify_model.purify(batch_adv, num_steps=20, lambda_schedule=schedule)
                
                purified_pred = classifier(purified).argmax(dim=1)
                correct_purified += (purified_pred == batch_labels).sum().item()
                
                conf_after = classifier.get_confidence(purified)
                confidence_improvements.extend((conf_after - conf_before).cpu().numpy())
                
                total += batch_labels.size(0)
        
        results[schedule] = {
            'clean_accuracy': correct_clean / total,
            'adversarial_accuracy': correct_adv / total,
            'purified_accuracy': correct_purified / total,
            'confidence_improvement': np.mean(confidence_improvements)
        }
    
    return results

def analyze_lambda_strategies(purify_model, adv_examples, device):
    """Analyze different lambda scheduling strategies."""
    purify_model.diffusion_model.eval()
    
    batch_size = 4
    sample_batch = adv_examples[:batch_size].to(device)
    
    strategies = ['exponential', 'polynomial', 'sigmoid']
    lambda_histories = {}
    confidence_histories = {}
    
    for strategy in strategies:
        print(f"Analyzing {strategy} strategy...")
        
        with torch.no_grad():
            _, conf_hist, lambda_hist = purify_model.purify(
                sample_batch, num_steps=30, lambda_schedule=strategy
            )
        
        lambda_histories[strategy] = np.array(lambda_hist)
        confidence_histories[strategy] = np.array(conf_hist)
    
    return lambda_histories, confidence_histories

def temporal_confidence_analysis(purify_model, adv_examples, device):
    """Analyze confidence evolution during purification process."""
    purify_model.diffusion_model.eval()
    
    batch_size = 4
    sample_batch = adv_examples[:batch_size].to(device)
    
    num_steps = 50
    confidence_evolution = []
    lambda_evolution = []
    
    print("Performing temporal analysis...")
    
    with torch.no_grad():
        _, conf_hist, lambda_hist = purify_model.purify(
            sample_batch, num_steps=num_steps, lambda_schedule='exponential'
        )
    
    confidence_evolution = np.array(conf_hist)
    lambda_evolution = np.array(lambda_hist)
    
    return confidence_evolution, lambda_evolution

def calculate_metrics(results):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    for schedule, result in results.items():
        metrics[schedule] = {
            'robustness_improvement': result['purified_accuracy'] - result['adversarial_accuracy'],
            'clean_degradation': result['clean_accuracy'] - result['purified_accuracy'],
            'overall_score': result['purified_accuracy'] + result['confidence_improvement']
        }
    
    return metrics
