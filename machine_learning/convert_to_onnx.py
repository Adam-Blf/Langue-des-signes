"""Model conversion script: scikit-learn → ONNX for GPU inference.

Converts trained RandomForest models to ONNX format for GPU-accelerated inference.
"""

import argparse
import joblib
import numpy as np
from pathlib import Path
from typing import Optional


def convert_model_to_onnx(
    sklearn_model_path: str,
    onnx_output_path: str,
    num_features: int = 63,
    test_inference: bool = True
) -> bool:
    """
    Convert scikit-learn model to ONNX format.
    
    Args:
        sklearn_model_path: Path to pickled sklearn model
        onnx_output_path: Path for output ONNX model
        num_features: Number of input features (21 landmarks × 3 = 63)
        test_inference: Whether to test inference after conversion
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import required libraries
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        print(f"\n{'='*70}")
        print("SKLEARN → ONNX MODEL CONVERSION")
        print(f"{'='*70}")
        
        # Load sklearn model
        print(f"\n1️⃣ Loading sklearn model from: {sklearn_model_path}")
        model = joblib.load(sklearn_model_path)
        print(f"   ✓ Model loaded: {type(model).__name__}")
        
        # Define input shape
        initial_type = [('float_input', FloatTensorType([None, num_features]))]
        print(f"\n2️⃣ Configuring ONNX conversion...")
        print(f"   Input shape: [batch_size, {num_features}]")
        
        # Convert to ONNX
        print(f"\n3️⃣ Converting to ONNX format...")
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12  # Compatible opset version
        )
        print(f"   ✓ Conversion successful")
        
        # Save ONNX model
        print(f"\n4️⃣ Saving ONNX model to: {onnx_output_path}")
        output_path = Path(onnx_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   ✓ Model saved ({file_size:.2f} MB)")
        
        # Test inference
        if test_inference:
            print(f"\n5️⃣ Testing inference...")
            success = test_onnx_inference(str(output_path), num_features)
            if success:
                print(f"   ✓ Inference test passed")
            else:
                print(f"   ⚠ Inference test failed (but model saved)")
        
        print(f"\n{'='*70}")
        print("✓ CONVERSION COMPLETE")
        print(f"{'='*70}")
        print(f"\nONNX model ready for GPU inference:")
        print(f"  {output_path}")
        print(f"\nUsage example:")
        print(f"  from gpu_inference import InferenceEngine")
        print(f"  engine = InferenceEngine()")
        print(f"  engine.load_model('{output_path}')")
        print(f"  letter, conf = engine.predict(features)")
        print(f"{'='*70}\n")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Error: Missing dependencies")
        print(f"   {e}")
        print(f"\nInstall required packages:")
        print(f"   pip install skl2onnx onnxruntime")
        return False
    
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_onnx_inference(onnx_model_path: str, num_features: int) -> bool:
    """
    Test ONNX model inference.
    
    Args:
        onnx_model_path: Path to ONNX model
        num_features: Number of input features
        
    Returns:
        True if inference works, False otherwise
    """
    try:
        import onnxruntime as ort
        
        # Create session
        session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Test with random data
        test_input = np.random.rand(1, num_features).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: test_input})
        
        # Verify output
        if outputs and len(outputs) > 0:
            prediction = outputs[0]
            return True
        
        return False
        
    except Exception as e:
        print(f"   Inference test error: {e}")
        return False


def benchmark_models(
    sklearn_model_path: str,
    onnx_model_path: str,
    num_samples: int = 1000
) -> None:
    """
    Benchmark sklearn vs ONNX inference speed.
    
    Args:
        sklearn_model_path: Path to sklearn model
        onnx_model_path: Path to ONNX model
        num_samples: Number of samples for benchmark
    """
    try:
        import time
        import onnxruntime as ort
        
        print(f"\n{'='*70}")
        print("PERFORMANCE BENCHMARK")
        print(f"{'='*70}")
        
        # Load models
        print(f"\nLoading models...")
        sklearn_model = joblib.load(sklearn_model_path)
        onnx_session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Prepare test data
        test_data = np.random.rand(num_samples, 63).astype(np.float32)
        print(f"Test data: {num_samples} samples × 63 features")
        
        # Benchmark sklearn
        print(f"\n1️⃣ Sklearn RandomForest (CPU):")
        start = time.time()
        for i in range(num_samples):
            sklearn_model.predict(test_data[i:i+1])
        sklearn_time = time.time() - start
        sklearn_fps = num_samples / sklearn_time
        print(f"   Time: {sklearn_time:.3f}s")
        print(f"   FPS: {sklearn_fps:.1f}")
        print(f"   Per sample: {(sklearn_time/num_samples)*1000:.2f}ms")
        
        # Benchmark ONNX
        print(f"\n2️⃣ ONNX Runtime (CPU):")
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        
        start = time.time()
        for i in range(num_samples):
            onnx_session.run([output_name], {input_name: test_data[i:i+1]})
        onnx_time = time.time() - start
        onnx_fps = num_samples / onnx_time
        print(f"   Time: {onnx_time:.3f}s")
        print(f"   FPS: {onnx_fps:.1f}")
        print(f"   Per sample: {(onnx_time/num_samples)*1000:.2f}ms")
        
        # Batch inference (ONNX advantage)
        print(f"\n3️⃣ ONNX Runtime (Batch {num_samples}):")
        start = time.time()
        onnx_session.run([output_name], {input_name: test_data})
        batch_time = time.time() - start
        batch_fps = num_samples / batch_time
        print(f"   Time: {batch_time:.3f}s")
        print(f"   FPS: {batch_fps:.1f}")
        print(f"   Per sample: {(batch_time/num_samples)*1000:.2f}ms")
        
        # Summary
        print(f"\n{'='*70}")
        print("SPEEDUP:")
        print(f"   ONNX vs Sklearn: {sklearn_time/onnx_time:.2f}x")
        print(f"   ONNX Batch vs Sklearn: {sklearn_time/batch_time:.2f}x")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert scikit-learn model to ONNX for GPU inference"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='machine_learning/model.pkl',
        help='Input sklearn model path (default: machine_learning/model.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='machine_learning/model.onnx',
        help='Output ONNX model path (default: machine_learning/model.onnx)'
    )
    parser.add_argument(
        '--features',
        type=int,
        default=63,
        help='Number of input features (default: 63)'
    )
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip inference test after conversion'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark after conversion'
    )
    parser.add_argument(
        '--benchmark-samples',
        type=int,
        default=1000,
        help='Number of samples for benchmark (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Convert model
    success = convert_model_to_onnx(
        sklearn_model_path=args.input,
        onnx_output_path=args.output,
        num_features=args.features,
        test_inference=not args.no_test
    )
    
    if not success:
        print("\n❌ Conversion failed")
        return 1
    
    # Optional benchmark
    if args.benchmark:
        benchmark_models(
            sklearn_model_path=args.input,
            onnx_model_path=args.output,
            num_samples=args.benchmark_samples
        )
    
    return 0


if __name__ == "__main__":
    exit(main())
