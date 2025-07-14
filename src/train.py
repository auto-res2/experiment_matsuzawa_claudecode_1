import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm

class SimpleClassifier(nn.Module):
    """Lightweight classifier for CIFAR-10 compatible with Tesla T4."""
    def __init__(self, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_confidence(self, x):
        """Get classifier confidence (max softmax probability)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, _ = torch.max(probs, dim=1)
            return confidence

class VEDiffusionModel(nn.Module):
    """Variance Exploding Diffusion Model for Purify-C."""
    def __init__(self, channels=3, time_dim=128):
        super(VEDiffusionModel, self).__init__()
        self.time_dim = time_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels + time_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )
        
    def forward(self, x, t):
        """Forward pass with time embedding."""
        t_embed = self.time_embed(t.view(-1, 1))
        t_embed = t_embed.view(t_embed.shape[0], self.time_dim, 1, 1)
        t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])
        
        x_t = torch.cat([x, t_embed], dim=1)
        return self.encoder(x_t)

class PurifyCModel:
    """Purify-C: Confidence-Guided Diffusion Purification."""
    def __init__(self, classifier, diffusion_model, device, sigma_min=0.01, sigma_max=50.0):
        self.classifier = classifier
        self.diffusion_model = diffusion_model
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def get_sigma_schedule(self, num_steps):
        """Generate sigma schedule for VE diffusion."""
        return torch.exp(torch.linspace(
            math.log(self.sigma_min), math.log(self.sigma_max), num_steps
        )).to(self.device)
    
    def lambda_exponential(self, confidence, lambda_0=1.0, alpha=2.0):
        """Exponential lambda scheduling based on confidence."""
        return lambda_0 * torch.exp(-alpha * confidence)
    
    def lambda_polynomial(self, confidence, lambda_0=1.0, power=2.0):
        """Polynomial lambda scheduling based on confidence."""
        return lambda_0 * (1 - confidence) ** power
    
    def lambda_sigmoid(self, confidence, lambda_0=1.0, steepness=10.0, midpoint=0.5):
        """Sigmoid lambda scheduling based on confidence."""
        return lambda_0 / (1 + torch.exp(steepness * (confidence - midpoint)))
    
    def get_confidence_gradient(self, x, target_class=None):
        """Compute gradient of log confidence for guidance."""
        with torch.no_grad():
            logits = self.classifier(x)
            probs = F.softmax(logits, dim=1)
            
            if target_class is None:
                confidence, target_class = torch.max(probs, dim=1)
            else:
                confidence = probs.gather(1, target_class.view(-1, 1)).squeeze()
        
        grad_approx = torch.randn_like(x) * 0.01 * (1 - confidence.view(-1, 1, 1, 1))
        
        return grad_approx, confidence
    
    def heun_step(self, x, t, dt, lambda_schedule='exponential'):
        """Heun's method step for numerical integration."""
        sigma = self.get_sigma_schedule(1)[0] * t.mean()  # Use mean of t for scalar sigma
        sigma = sigma.view(1, 1, 1, 1)  # Reshape for broadcasting
        
        with torch.no_grad():
            score = self.diffusion_model(x, t)
        
        confidence_grad, confidence = self.get_confidence_gradient(x)
        
        if lambda_schedule == 'exponential':
            lambda_t = self.lambda_exponential(confidence)
        elif lambda_schedule == 'polynomial':
            lambda_t = self.lambda_polynomial(confidence)
        elif lambda_schedule == 'sigmoid':
            lambda_t = self.lambda_sigmoid(confidence)
        else:
            lambda_t = torch.ones_like(confidence)
        
        lambda_t = lambda_t.view(-1, 1, 1, 1)
        
        f_t = -0.5 * sigma**2 * score + lambda_t * confidence_grad
        noise = torch.randn_like(x) * sigma * math.sqrt(dt)
        
        x_pred = x + f_t * dt + noise
        x_pred = torch.clamp(x_pred, 0, 1)
        
        t_pred = torch.clamp(t - dt, min=0.0)
        with torch.no_grad():
            score_pred = self.diffusion_model(x_pred, t_pred)
        
        confidence_grad_pred, confidence_pred = self.get_confidence_gradient(x_pred)
        
        if lambda_schedule == 'exponential':
            lambda_pred = self.lambda_exponential(confidence_pred)
        elif lambda_schedule == 'polynomial':
            lambda_pred = self.lambda_polynomial(confidence_pred)
        elif lambda_schedule == 'sigmoid':
            lambda_pred = self.lambda_sigmoid(confidence_pred)
        else:
            lambda_pred = torch.ones_like(confidence_pred)
        
        lambda_pred = lambda_pred.view(-1, 1, 1, 1)
        
        f_pred = -0.5 * sigma**2 * score_pred + lambda_pred * confidence_grad_pred
        
        x_next = x + 0.5 * (f_t + f_pred) * dt + noise
        x_next = torch.clamp(x_next, 0, 1)
        
        return x_next, confidence, lambda_t.squeeze()
    
    def purify(self, x_adv, num_steps=50, lambda_schedule='exponential'):
        """Purify adversarial examples using confidence-guided diffusion."""
        x = x_adv.clone()
        dt = 1.0 / num_steps
        
        confidence_history = []
        lambda_history = []
        
        for step in range(num_steps):
            t = torch.ones(x.shape[0], device=self.device) * (1.0 - step * dt)
            x, confidence, lambda_t = self.heun_step(x, t, dt, lambda_schedule)
            
            confidence_history.append(confidence.cpu().numpy())
            lambda_history.append(lambda_t.cpu().numpy())
        
        return x, confidence_history, lambda_history

def train_classifier(trainloader, device, epochs=5):
    """Train a simple classifier for CIFAR-10."""
    classifier = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    
    classifier.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    return classifier

def train_diffusion_model(trainloader, device, epochs=3):
    """Train diffusion model for denoising."""
    diffusion_model = VEDiffusionModel().to(device)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=0.0001)
    
    diffusion_model.train()
    for epoch in range(epochs):
        for i, (inputs, _) in enumerate(tqdm(trainloader, desc=f"Diffusion Epoch {epoch+1}")):
            inputs = inputs.to(device)
            
            t = torch.rand(inputs.shape[0], device=device)
            noise = torch.randn_like(inputs)
            sigma = 0.01 + (50.0 - 0.01) * t.view(-1, 1, 1, 1)
            
            noisy_inputs = inputs + sigma * noise
            
            optimizer.zero_grad()
            predicted_noise = diffusion_model(noisy_inputs, t)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            if i % 50 == 49:
                print(f'Diffusion loss: {loss.item():.4f}')
    
    return diffusion_model
