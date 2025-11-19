"""GPU-accelerated inference module using PyTorch/ONNX.

Provides faster inference for sign language detection using GPU when available.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class InferenceBackend(Enum):
    """Available inference backends."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    SKLEARN = "sklearn"  # Fallback to scikit-learn


@dataclass
class InferenceConfig:
    """Configuration for GPU inference."""
    
    backend: InferenceBackend = InferenceBackend.PYTORCH
    use_gpu: bool = True
    model_path: Optional[str] = None
    batch_size: int = 1
    num_threads: int = 4
    
    # Performance optimization
    use_fp16: bool = False  # Half precision for faster inference
    use_tensorrt: bool = False  # NVIDIA TensorRT optimization


class GPUDetector:
    """Detects available GPU and capabilities."""
    
    @staticmethod
    def has_cuda() -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @staticmethod
    def has_mps() -> bool:
        """Check if Apple Metal Performance Shaders is available."""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            return False
    
    @staticmethod
    def has_onnx_gpu() -> bool:
        """Check if ONNX Runtime GPU is available."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
        except ImportError:
            return False
    
    @staticmethod
    def get_device_info() -> dict:
        """Get detailed GPU device information."""
        info = {
            "has_cuda": GPUDetector.has_cuda(),
            "has_mps": GPUDetector.has_mps(),
            "has_onnx_gpu": GPUDetector.has_onnx_gpu(),
            "recommended_backend": None,
            "device_name": None,
            "compute_capability": None
        }
        
        if info["has_cuda"]:
            try:
                import torch
                info["device_name"] = torch.cuda.get_device_name(0)
                info["compute_capability"] = torch.cuda.get_device_capability(0)
                info["recommended_backend"] = "pytorch"
            except Exception as e:
                print(f"Error getting CUDA info: {e}")
        
        elif info["has_mps"]:
            info["device_name"] = "Apple Metal"
            info["recommended_backend"] = "pytorch"
        
        elif info["has_onnx_gpu"]:
            info["recommended_backend"] = "onnx"
        
        else:
            info["recommended_backend"] = "sklearn"
        
        return info
    
    @staticmethod
    def print_device_info():
        """Print GPU device information."""
        info = GPUDetector.get_device_info()
        
        print("\n" + "="*60)
        print("GPU Device Information")
        print("="*60)
        
        print(f"\n🔹 CUDA (NVIDIA): {'✅ Available' if info['has_cuda'] else '❌ Not available'}")
        print(f"🔹 MPS (Apple Metal): {'✅ Available' if info['has_mps'] else '❌ Not available'}")
        print(f"🔹 ONNX GPU: {'✅ Available' if info['has_onnx_gpu'] else '❌ Not available'}")
        
        if info['device_name']:
            print(f"\n📱 Device: {info['device_name']}")
        
        if info['compute_capability']:
            print(f"⚡ Compute Capability: {'.'.join(map(str, info['compute_capability']))}")
        
        print(f"\n💡 Recommended Backend: {info['recommended_backend'].upper()}\n")


class PyTorchInference:
    """PyTorch-based GPU inference engine."""
    
    def __init__(self, config: InferenceConfig):
        """Initialize PyTorch inference engine."""
        self.config = config
        self.model = None
        self.device = self._get_device()
        
        print(f"PyTorch inference initialized on device: {self.device}")
    
    def _get_device(self) -> str:
        """Determine best available device."""
        try:
            import torch
            
            if self.config.use_gpu:
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            
            return "cpu"
        
        except ImportError:
            return "cpu"
    
    def load_model(self, model_path: str):
        """Load PyTorch model from file."""
        try:
            import torch
            
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.eval()
            
            # Apply optimizations
            if self.config.use_fp16 and self.device in ("cuda", "mps"):
                self.model = self.model.half()
            
            # Compile for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
            
            print(f"Model loaded: {model_path}")
            
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Run inference on input features.
        
        Args:
            features: Input feature array (63 landmarks)
        
        Returns:
            Tuple of (predicted_letter, confidence)
        """
        try:
            import torch
            
            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            if self.config.use_fp16 and self.device in ("cuda", "mps"):
                x = x.half()
            
            x = x.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(x)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)
            
            # Convert to letter
            letter = chr(65 + predicted.item())  # 0=A, 1=B, etc.
            conf = confidence.item()
            
            return letter, conf
        
        except Exception as e:
            print(f"Inference error: {e}")
            return "?", 0.0
    
    def predict_batch(self, features_batch: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Run batch inference."""
        try:
            import torch
            
            # Convert batch to tensor
            x = torch.tensor(np.array(features_batch), dtype=torch.float32)
            
            if self.config.use_fp16 and self.device in ("cuda", "mps"):
                x = x.half()
            
            x = x.to(self.device)
            
            # Run batch inference
            with torch.no_grad():
                outputs = self.model(x)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
            
            # Convert to letters
            results = []
            for pred, conf in zip(predictions, confidences):
                letter = chr(65 + pred.item())
                results.append((letter, conf.item()))
            
            return results
        
        except Exception as e:
            print(f"Batch inference error: {e}")
            return [("?", 0.0)] * len(features_batch)


class ONNXInference:
    """ONNX Runtime GPU inference engine."""
    
    def __init__(self, config: InferenceConfig):
        """Initialize ONNX inference engine."""
        self.config = config
        self.session = None
        self.input_name = None
        self.output_name = None
        
        print("ONNX inference initialized")
    
    def load_model(self, model_path: str):
        """Load ONNX model from file."""
        try:
            import onnxruntime as ort
            
            # Configure execution providers (GPU first, then CPU)
            providers = []
            
            if self.config.use_gpu:
                if self.config.use_tensorrt:
                    providers.append('TensorrtExecutionProvider')
                providers.append('CUDAExecutionProvider')
            
            providers.append('CPUExecutionProvider')
            
            # Create inference session
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.num_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=providers
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"ONNX model loaded: {model_path}")
            print(f"Execution providers: {self.session.get_providers()}")
            
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Run inference on input features."""
        try:
            # Reshape for batch
            x = features.reshape(1, -1).astype(np.float32)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: x})
            probabilities = outputs[0][0]
            
            # Get prediction
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            
            letter = chr(65 + predicted_idx)
            
            return letter, float(confidence)
        
        except Exception as e:
            print(f"Inference error: {e}")
            return "?", 0.0
    
    def predict_batch(self, features_batch: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Run batch inference."""
        try:
            x = np.array(features_batch, dtype=np.float32)
            
            outputs = self.session.run([self.output_name], {self.input_name: x})
            probabilities = outputs[0]
            
            results = []
            for probs in probabilities:
                predicted_idx = np.argmax(probs)
                confidence = probs[predicted_idx]
                letter = chr(65 + predicted_idx)
                results.append((letter, float(confidence)))
            
            return results
        
        except Exception as e:
            print(f"Batch inference error: {e}")
            return [("?", 0.0)] * len(features_batch)


class InferenceEngine:
    """Unified inference engine with automatic backend selection."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize inference engine.
        
        Args:
            config: Inference configuration (auto-configured if None)
        """
        if config is None:
            config = self._auto_configure()
        
        self.config = config
        self.backend = None
        
        # Initialize appropriate backend
        if config.backend == InferenceBackend.PYTORCH:
            self.backend = PyTorchInference(config)
        elif config.backend == InferenceBackend.ONNX:
            self.backend = ONNXInference(config)
        else:
            # Fallback to scikit-learn
            print("Using scikit-learn backend (CPU only)")
    
    def _auto_configure(self) -> InferenceConfig:
        """Auto-configure inference based on available hardware."""
        device_info = GPUDetector.get_device_info()
        
        if device_info["recommended_backend"] == "pytorch":
            return InferenceConfig(
                backend=InferenceBackend.PYTORCH,
                use_gpu=True,
                use_fp16=device_info["has_cuda"]  # FP16 only on CUDA
            )
        elif device_info["recommended_backend"] == "onnx":
            return InferenceConfig(
                backend=InferenceBackend.ONNX,
                use_gpu=True
            )
        else:
            return InferenceConfig(
                backend=InferenceBackend.SKLEARN,
                use_gpu=False
            )
    
    def load_model(self, model_path: str):
        """Load model from file."""
        if self.backend:
            self.backend.load_model(model_path)
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Run inference."""
        if self.backend:
            return self.backend.predict(features)
        return "?", 0.0
    
    def predict_batch(self, features_batch: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Run batch inference."""
        if self.backend:
            return self.backend.predict_batch(features_batch)
        return [("?", 0.0)] * len(features_batch)


def convert_sklearn_to_pytorch(sklearn_model_path: str, output_path: str, num_features: int = 63, num_classes: int = 26):
    """Convert scikit-learn RandomForest to PyTorch model."""
    try:
        import torch
        import torch.nn as nn
        import joblib
        
        # Load scikit-learn model
        sklearn_model = joblib.load(sklearn_model_path)
        
        # Create simple PyTorch model
        class SimpleClassifier(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                return self.network(x)
        
        model = SimpleClassifier(num_features, num_classes)
        
        # Save model
        torch.save(model, output_path)
        print(f"PyTorch model saved: {output_path}")
        print("Note: Model architecture created, but weights not trained.")
        print("Train this model on your dataset for best results.")
        
    except Exception as e:
        print(f"Error converting model: {e}")


def convert_sklearn_to_onnx(sklearn_model_path: str, output_path: str, num_features: int = 63):
    """Convert scikit-learn model to ONNX format."""
    try:
        import joblib
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Load scikit-learn model
        model = joblib.load(sklearn_model_path)
        
        # Define input shape
        initial_type = [('float_input', FloatTensorType([None, num_features]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Save ONNX model
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"ONNX model saved: {output_path}")
        
    except Exception as e:
        print(f"Error converting to ONNX: {e}")
        print("Install required packages: pip install skl2onnx onnxruntime")


if __name__ == "__main__":
    # Demo usage
    print("GPU Inference Engine Demo")
    print("=" * 60)
    
    # Check GPU availability
    GPUDetector.print_device_info()
    
    # Create inference engine with auto-configuration
    engine = InferenceEngine()
    
    print(f"Backend: {engine.config.backend.value}")
    print(f"GPU enabled: {engine.config.use_gpu}")
    
    # Demo inference with random features
    print("\nDemo inference with random features...")
    random_features = np.random.rand(63).astype(np.float32)
    
    # Note: This will fail without a loaded model
    # letter, confidence = engine.predict(random_features)
    # print(f"Predicted: {letter} (confidence: {confidence:.2f})")
    
    print("\n💡 To use GPU inference:")
    print("1. Convert your scikit-learn model:")
    print("   convert_sklearn_to_onnx('model.pkl', 'model.onnx')")
    print("2. Load and use:")
    print("   engine.load_model('model.onnx')")
    print("   letter, conf = engine.predict(features)")
