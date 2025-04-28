import torch
import torch.onnx
import argparse
from pathlib import Path
from typing import Optional, Union, Dict

def load_model(
    model_path: Union[str, Path], 
    model_type: str, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.nn.Module:
    """
    Load a checkpoint file into the specified model architecture.
    
    Args:
        model_path: Path to the checkpoint file
        model_type: Type of model to load ('vit', 'cait', 'swin', etc.)
        device: Device to load the model onto
        
    Returns:
        Loaded model
    """
    from models.vit import ViT
    from models.cait import CaiT
    from models.swin import swin_t
    
    # Initialize model architecture based on type
    if model_type == "vit":
        model = ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif model_type == "cait":
        model = CaiT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=512,
            depth=6,
            cls_depth=2,
            heads=8,
            mlp_dim=512
        )
    elif model_type == "swin":
        model = swin_t(
            window_size=4,
            num_classes=10,
            downscaling_factors=(2,2,2,1)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def export_to_onnx(
    model: torch.nn.Module,
    save_path: Union[str, Path],
    input_shape: tuple = (1, 3, 32, 32),
    dynamic_axes: Optional[Dict] = None
) -> None:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: The model to export
        save_path: Where to save the ONNX model
        input_shape: Input tensor shape
        dynamic_axes: Dynamic axes configuration for ONNX export
    """
    dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
    
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=12,
        export_params=True,
        do_constant_folding=True
    )
    print(f"ONNX model saved to {save_path}")

def export_to_torchscript(
    model: torch.nn.Module,
    save_path: Union[str, Path],
    input_shape: tuple = (1, 3, 32, 32),
    use_trace: bool = True
) -> None:
    """
    Export a PyTorch model to TorchScript format.
    
    Args:
        model: The model to export
        save_path: Where to save the TorchScript model
        input_shape: Input tensor shape
        use_trace: Whether to use tracing (True) or scripting (False)
    """
    if use_trace:
        dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
        traced_model = torch.jit.trace(model, dummy_input)
        script_model = traced_model
    else:
        script_model = torch.jit.script(model)
    
    script_model.save(save_path)
    print(f"TorchScript model saved to {save_path}")

def verify_exports(
    original_model: torch.nn.Module,
    onnx_path: Union[str, Path],
    torchscript_path: Union[str, Path],
    input_shape: tuple = (1, 3, 32, 32)
) -> None:
    """
    Verify that the exported models produce the same outputs as the original model.
    
    Args:
        original_model: The original PyTorch model
        onnx_path: Path to exported ONNX model
        torchscript_path: Path to exported TorchScript model
        input_shape: Input tensor shape for testing
    """
    import onnxruntime
    
    # Generate test input
    device = next(original_model.parameters()).device
    test_input = torch.randn(input_shape, device=device)
    
    # Get original model prediction
    with torch.no_grad():
        original_output = original_model(test_input)
    
    # Verify TorchScript model
    ts_model = torch.jit.load(torchscript_path)
    with torch.no_grad():
        ts_output = ts_model(test_input)
    
    # Verify ONNX model
    ort_session = onnxruntime.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
    ort_output = torch.tensor(ort_session.run(None, ort_inputs)[0])
    
    # Compare outputs
    torch.testing.assert_close(original_output.cpu(), ts_output.cpu(), rtol=1e-03, atol=1e-03)
    torch.testing.assert_close(original_output.cpu(), ort_output, rtol=1e-03, atol=1e-03)
    print("Export verification successful! All outputs match within tolerance.")

def main():
    parser = argparse.ArgumentParser(description='Export Vision Transformer models to ONNX and TorchScript')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True, choices=['vit', 'cait', 'swin'], 
                        help='Type of model architecture')
    parser.add_argument('--output_dir', type=str, default='exported_models', 
                        help='Directory to save exported models')
    parser.add_argument('--img_size', type=int, default=32, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--verify', action='store_true', help='Verify exported models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, args.model_type)
    
    # Define input shape
    input_shape = (args.batch_size, 3, args.img_size, args.img_size)
    
    # Export models
    onnx_path = output_dir / f"{args.model_type}.onnx"
    torchscript_path = output_dir / f"{args.model_type}.pt"
    
    export_to_onnx(model, onnx_path, input_shape)
    export_to_torchscript(model, torchscript_path, input_shape)
    
    if args.verify:
        verify_exports(model, onnx_path, torchscript_path, input_shape)

if __name__ == "__main__":
    main()