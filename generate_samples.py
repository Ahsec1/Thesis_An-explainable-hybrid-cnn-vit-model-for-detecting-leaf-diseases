# generate_samples.py
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import cv2
from tqdm import tqdm
from collections import defaultdict
import shap
import io

# ====== EXPLANATION CODE INTEGRATION ======
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        try:
            self.target_layer.register_full_backward_hook(backward_hook)
        except Exception:
            self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        output.backward(gradient=one_hot, retain_graph=False)

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)
        weighted = (pooled_grads.unsqueeze(1).unsqueeze(1) * activations).sum(dim=0)
        
        heatmap = weighted.cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / (heatmap.max() + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)
        return heatmap

def compute_attention_rollout_vit_small(model, input_tensor):
    """Compute attention rollout for ViT models"""
    model.eval()
    device_model = next(model.parameters()).device
    attn_matrices = []

    def make_hook(block):
        def hook(module, input, output):
            x = input[0]
            if not hasattr(module, "qkv"):
                return
            qkv = module.qkv(x)
            B, N, _ = qkv.shape
            num_heads = getattr(module, "num_heads", 8)
            head_dim = getattr(module, "head_dim", _ // (3 * num_heads))
            
            qkv_reshaped = qkv.reshape(B, N, 3, num_heads, head_dim)
            qkv_reshaped = qkv_reshaped.permute(2, 0, 3, 1, 4)
            q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]
            
            attn = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))
            attn = torch.softmax(attn, dim=-1)
            attn_matrices.append(attn.detach().cpu())
        return hook

    hooks = []
    for blk in getattr(model, "blocks", []):
        if hasattr(blk, "attn"):
            hooks.append(blk.attn.register_forward_hook(make_hook(blk)))

    with torch.no_grad():
        _ = model(input_tensor.to(device_model))

    for h in hooks:
        h.remove()

    if len(attn_matrices) == 0:
        side = int((model.patch_embed.num_patches) ** 0.5) if hasattr(model, "patch_embed") else 14
        return np.zeros((side, side))

    attn = torch.stack(attn_matrices).squeeze(1).mean(dim=1)
    num_tokens = attn.size(-1)
    result = torch.eye(num_tokens)
    for layer_attn in attn:
        a = layer_attn + torch.eye(num_tokens)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    mask = result[0, 1:]
    side = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(side, side).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask

def find_last_conv_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("❌ No Conv2D layer found in model!")

def overlay_heatmap(img, heatmap):
    """Overlay heatmap on image"""
    hmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hmap_resized = np.clip(hmap_resized, 0, 1)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * hmap_resized), cv2.COLORMAP_JET)
    
    if img.dtype != np.uint8:
        base = (img * 255).astype(np.uint8)
    else:
        base = img.copy()
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(base_bgr, 0.6, heatmap_color, 0.4, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return blended_rgb

def fig_to_array(fig):
    """Convert matplotlib figure to numpy array"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

# --- ENSEMBLE SHAP GENERATION ---
class EnsembleWrapper(torch.nn.Module):
    """Wrapper to create ensemble model for SHAP"""
    def __init__(self, models_dict, weights_dict):
        super().__init__()
        self.models = torch.nn.ModuleDict({k: v for k, v in models_dict.items()})
        self.weights = weights_dict

    def forward(self, x):
        out = None
        for name, model in self.models.items():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            weighted = probs * self.weights[name]
            out = weighted if out is None else out + weighted
        return out

def generate_ensemble_shap(ensemble_model, image_tensor, background_data, 
                           transform_inv, test_dataset, device):
    """
    Generate SHAP explanation for ensemble model.
    Returns 5-panel visualization as numpy array.
    """
    try:
        ensemble_model.eval()
        
        image_tensor = image_tensor.to(device)
        background_data = background_data.to(device)
        
        # Build explainer
        explainer = shap.GradientExplainer(ensemble_model, background_data)
        
        # Get SHAP values
        shap_values = explainer.shap_values(image_tensor.unsqueeze(0))
        
        # Get prediction
        with torch.no_grad():
            probs = ensemble_model(image_tensor.unsqueeze(0))
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Extract SHAP for predicted class
        shap_for_pred = shap_values[0, :, :, :, pred_class]  # (3, 224, 224)
        
        # Get image
        img_pil = transform_inv(image_tensor.cpu())
        img_array = np.array(img_pil)
        
        # Calculate SHAP components
        shap_abs = np.mean(np.abs(shap_for_pred), axis=0)
        shap_positive = np.mean(np.maximum(shap_for_pred, 0), axis=0)
        shap_negative = np.mean(np.maximum(-shap_for_pred, 0), axis=0)
        
        vmax_abs = np.percentile(shap_abs, 99) or 1e-8
        vmax_pos = np.percentile(shap_positive, 99) or 1e-8
        vmax_neg = np.percentile(shap_negative, 99) if shap_negative.max() > 0 else 1e-8
        
        # Create overlay
        hmap = cv2.resize(shap_abs, (img_array.shape[1], img_array.shape[0]))
        if hmap.max() > hmap.min():
            hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
        else:
            hmap = np.zeros_like(hmap)
        
        # Create 5-panel figure
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        true_label = test_dataset.classes[test_dataset.class_to_idx[image_tensor]] if hasattr(test_dataset, 'class_to_idx') else "Unknown"
        pred_name = test_dataset.classes[pred_class]
        correct = "✅" if pred_class == pred_class else "❌"  # You may need to pass true label

        fig.suptitle(
            f"Pred: {pred_name} (conf: {confidence:.3f})",
            fontsize=11, fontweight='bold'
        )

        # Panel 1: Original
        axes[0].imshow(img_array)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Panel 2: SHAP Absolute Importance
        im1 = axes[1].imshow(shap_abs, cmap='hot', vmin=0, vmax=vmax_abs)
        axes[1].set_title("SHAP |Importance|")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Positive Contributions
        im2 = axes[2].imshow(shap_positive, cmap='Reds', vmin=0, vmax=vmax_pos)
        axes[2].set_title("Positive")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4: Negative Contributions
        im3 = axes[3].imshow(shap_negative, cmap='Blues', vmin=0, vmax=vmax_neg)
        axes[3].set_title("Negative")
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        # Panel 5: Overlay
        axes[4].imshow(img_array)
        axes[4].imshow(hmap, cmap='hot', alpha=0.6)
        axes[4].set_title("Overlay")
        axes[4].axis("off")

        plt.tight_layout()
        
        # Convert to array
        shap_image = fig_to_array(fig)
        plt.close(fig)
        
        return shap_image
        
    except Exception as e:
        print(f"❌ Ensemble SHAP error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_explanations(models, image_tensor, predicted_class, device, 
                         background_data=None, ensemble_weights=None):
    """Generate all explanations for a single image"""
    explanations = {}
    
    img_denormalized = denormalize_image(image_tensor.cpu())
    img_pil = tensor_to_pil(img_denormalized)
    img_np = np.array(img_pil)
    
    # Generate Grad-CAM explanations for individual models
    try:
        last_conv = find_last_conv_layer(models['resnet50'])
        gradcam_resnet = GradCAM(models['resnet50'], last_conv)
        heatmap_resnet = gradcam_resnet.generate(image_tensor.unsqueeze(0).to(device), class_idx=predicted_class)
        overlay_resnet = overlay_heatmap(img_np, heatmap_resnet)
        explanations['resnet50_gradcam'] = {
            'heatmap': heatmap_resnet,
            'overlay': overlay_resnet
        }
    except Exception as e:
        print(f"❌ ResNet50 Grad-CAM error: {e}")
        explanations['resnet50_gradcam'] = None

    try:
        last_conv = find_last_conv_layer(models['efficientnet_b0'])
        gradcam_eff = GradCAM(models['efficientnet_b0'], last_conv)
        heatmap_eff = gradcam_eff.generate(image_tensor.unsqueeze(0).to(device), class_idx=predicted_class)
        overlay_eff = overlay_heatmap(img_np, heatmap_eff)
        explanations['efficientnet_gradcam'] = {
            'heatmap': heatmap_eff,
            'overlay': overlay_eff
        }
    except Exception as e:
        print(f"❌ EfficientNet Grad-CAM error: {e}")
        explanations['efficientnet_gradcam'] = None

    try:
        with torch.no_grad():
            vit_attention = compute_attention_rollout_vit_small(
                models['vit_small'], 
                image_tensor.unsqueeze(0).to(device)
            )
        overlay_vit = overlay_heatmap(img_np, vit_attention)
        explanations['vit_attention'] = {
            'heatmap': vit_attention,
            'overlay': overlay_vit
        }
    except Exception as e:
        print(f"❌ ViT Attention error: {e}")
        explanations['vit_attention'] = None

    # Generate ENSEMBLE SHAP only (not individual models)
    if background_data is not None and ensemble_weights is not None:
        try:
            print("🔄 Generating Ensemble SHAP...")
            
            # Create ensemble wrapper
            ensemble_model = EnsembleWrapper(models, ensemble_weights).to(device)
            ensemble_model.eval()
            
            # Inverse normalization transform
            transform_inv = transforms.Compose([
                transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                ),
                transforms.ToPILImage()
            ])
            
            # Generate SHAP
            shap_image = generate_ensemble_shap(
                ensemble_model, image_tensor, background_data,
                transform_inv, test_dataset, device
            )
            
            if shap_image is not None:
                explanations['shap'] = {
                    'ensemble': shap_image  # Single 5-panel image
                }
                print("✅ Ensemble SHAP generated successfully")
            else:
                explanations['shap'] = None
                
        except Exception as e:
            print(f"❌ Ensemble SHAP error: {e}")
            explanations['shap'] = None

    return explanations

def generate_evaluation_samples(test_loader, models, test_dataset, num_samples=30, 
                                device=None, include_shap=True, ensemble_weights=None):
    """Generate samples with explanations for evaluation interface"""
    samples = []
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔄 Generating {num_samples} evaluation samples on {device}...")
    
    for name, model in models.items():
        model.to(device)
        model.eval()
    
    # Prepare background data for SHAP
    background_data = None
    if include_shap:
        print("📊 Preparing SHAP background data...")
        try:
            background_batch = next(iter(test_loader))
            background_images = background_batch[0][:10].to(device)  # Use 10 samples
            background_data = background_images
            print(f"✅ Prepared SHAP background with {len(background_images)} samples")
        except Exception as e:
            print(f"❌ Failed to prepare SHAP background: {e}")
            include_shap = False
    
    # Check for ensemble weights
    if ensemble_weights is None:
        print("⚠️  No ensemble_weights provided, SHAP will use equal weights")
        ensemble_weights = {name: 1.0/len(models) for name in models.keys()}
    
    class_counts = defaultdict(int)
    max_per_class = max(2, num_samples // len(test_dataset.classes))
    
    sample_count = 0
    pbar = tqdm(total=num_samples, desc="📊 Generating samples")
    
    balanced_samples = []
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        if len(balanced_samples) >= num_samples:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            predictions = {}
            ensemble_probs = []
            
            for name, model in models.items():
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                pred_classes = torch.argmax(outputs, dim=1)
                
                predictions[name] = {
                    'probabilities': probs.cpu().numpy(),
                    'predicted_class': pred_classes.cpu().numpy(),
                    'confidence': probs.max(dim=1)[0].cpu().numpy()
                }
                ensemble_probs.append(probs * ensemble_weights[name])
            
            # Weighted ensemble
            ensemble_probs = torch.stack(ensemble_probs).sum(dim=0)
            ensemble_pred = torch.argmax(ensemble_probs, dim=1)
            ensemble_confidence = ensemble_probs.max(dim=1)[0]
            
            for i in range(len(images)):
                if len(balanced_samples) >= num_samples:
                    break
                
                true_label = labels[i].item()
                true_class = test_dataset.classes[true_label]
                
                if class_counts[true_class] >= max_per_class:
                    continue
                
                class_counts[true_class] += 1
                
                sample_info = {
                    'image': images[i],
                    'true_label': true_class,
                    'true_label_idx': true_label,
                    'ensemble_pred_class': ensemble_pred[i].item(),
                    'ensemble_confidence': ensemble_confidence[i].item(),
                    'individual_predictions': {
                        name: {
                            'class': test_dataset.classes[pred['predicted_class'][i]],
                            'confidence': pred['confidence'][i],
                            'class_idx': pred['predicted_class'][i]
                        } for name, pred in predictions.items()
                    }
                }
                balanced_samples.append(sample_info)
    
    print("🔥 Generating explanations for balanced samples...")
    
    for i, sample_info in enumerate(balanced_samples):
        if sample_count >= num_samples:
            break
            
        explanations = generate_explanations(
            models, sample_info['image'], 
            sample_info['ensemble_pred_class'], device,
            background_data=background_data if include_shap else None,
            ensemble_weights=ensemble_weights
        )
        
        img_cpu = sample_info['image'].cpu()
        img_denormalized = denormalize_image(img_cpu)
        img_pil = tensor_to_pil(img_denormalized)
        
        sample = {
            'image_array': np.array(img_pil),
            'true_label': sample_info['true_label'],
            'true_label_idx': sample_info['true_label_idx'],
            'ensemble_prediction': {
                'class': test_dataset.classes[sample_info['ensemble_pred_class']],
                'confidence': sample_info['ensemble_confidence'],
                'class_idx': sample_info['ensemble_pred_class']
            },
            'individual_predictions': sample_info['individual_predictions'],
            'explanations': explanations,
            'sample_id': sample_count
        }
        samples.append(sample)
        sample_count += 1
        pbar.update(1)
    
    pbar.close()
    print(f"✅ Generated {len(samples)} evaluation samples")
    
    # Debug: Check SHAP structure
    print("\n🔍 Verifying SHAP data...")
    for i, sample in enumerate(samples):
        has_shap = 'shap' in sample.get('explanations', {}) and sample['explanations']['shap'] is not None
        if has_shap:
            shap_keys = list(sample['explanations']['shap'].keys())
            print(f"Sample {i}: SHAP keys: {shap_keys}")
        else:
            print(f"Sample {i}: No SHAP data")
    
    return samples

def denormalize_image(tensor):
    """Reverse the ImageNet normalization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    return Image.fromarray(np.uint8(tensor.numpy().transpose(1, 2, 0) * 255))

def verify_shap_structure(samples):
    """Verify the structure of SHAP data in generated samples"""
    print("\n🔍 Verifying SHAP data structure...")
    for i, sample in enumerate(samples):
        print(f"\nSample {i}:")
        
        if 'explanations' in sample and 'shap' in sample['explanations']:
            shap_data = sample['explanations']['shap']
            if shap_data is None:
                print("  SHAP: None")
            else:
                print(f"  SHAP keys: {list(shap_data.keys())}")
                for key, value in shap_data.items():
                    if isinstance(value, np.ndarray):
                        print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"    {key}: {type(value)}")
        else:
            print("  No SHAP data found")
    
    return True

# Test function
def test_sample_generation():
    print("🧪 Sample generator with ensemble SHAP ready!")