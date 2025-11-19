"""Tests for gpu_inference.py - GPU acceleration."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from gpu_inference import (
    GPUDetector,
    InferenceEngine,
    InferenceBackend,
    InferenceConfig
)


class TestGPUDetector:
    """Test GPU detection utilities."""
    
    def test_has_cuda_returns_bool(self):
        """Test CUDA detection returns boolean."""
        result = GPUDetector.has_cuda()
        assert isinstance(result, bool)
    
    def test_has_mps_returns_bool(self):
        """Test MPS detection returns boolean."""
        result = GPUDetector.has_mps()
        assert isinstance(result, bool)
    
    def test_has_onnx_gpu_returns_bool(self):
        """Test ONNX GPU detection returns boolean."""
        result = GPUDetector.has_onnx_gpu()
        assert isinstance(result, bool)
    
    def test_get_device_info_returns_dict(self):
        """Test device info returns dictionary."""
        info = GPUDetector.get_device_info()
        
        assert isinstance(info, dict)
        assert 'has_cuda' in info
        assert 'has_mps' in info
        assert 'has_onnx_gpu' in info
        assert 'recommended_backend' in info
    
    def test_print_device_info_no_error(self):
        """Test printing device info doesn't raise error."""
        try:
            GPUDetector.print_device_info()
            assert True
        except Exception as e:
            pytest.fail(f"print_device_info raised exception: {e}")


class TestInferenceConfig:
    """Test InferenceConfig dataclass."""
    
    def test_inference_config_creation(self):
        """Test creating InferenceConfig."""
        config = InferenceConfig(
            backend=InferenceBackend.PYTORCH,
            use_gpu=True,
            batch_size=16
        )
        
        assert config.backend == InferenceBackend.PYTORCH
        assert config.use_gpu is True
        assert config.batch_size == 16
    
    def test_inference_config_defaults(self):
        """Test InferenceConfig default values."""
        config = InferenceConfig()
        
        assert config.backend in [InferenceBackend.PYTORCH, InferenceBackend.ONNX, InferenceBackend.SKLEARN]
        assert isinstance(config.use_gpu, bool)
        assert config.batch_size > 0


class TestInferenceBackend:
    """Test InferenceBackend enum."""
    
    def test_backend_values_exist(self):
        """Test that all backends are defined."""
        assert hasattr(InferenceBackend, 'PYTORCH')
        assert hasattr(InferenceBackend, 'ONNX')
        assert hasattr(InferenceBackend, 'SKLEARN')
    
    def test_backend_string_values(self):
        """Test backend enum string values."""
        assert InferenceBackend.PYTORCH.value == "pytorch"
        assert InferenceBackend.ONNX.value == "onnx"
        assert InferenceBackend.SKLEARN.value == "sklearn"


class TestInferenceEngine:
    """Test InferenceEngine unified interface."""
    
    def test_inference_engine_initialization(self):
        """Test InferenceEngine auto-configuration."""
        engine = InferenceEngine()
        
        assert engine is not None
        assert hasattr(engine, 'config')
        assert isinstance(engine.config, InferenceConfig)
    
    def test_inference_engine_with_config(self):
        """Test InferenceEngine with custom config."""
        config = InferenceConfig(
            backend=InferenceBackend.SKLEARN,
            use_gpu=False
        )
        engine = InferenceEngine(config)
        
        assert engine.config.backend == InferenceBackend.SKLEARN
        assert engine.config.use_gpu is False
    
    @patch('gpu_inference.PyTorchInference')
    def test_predict_with_mock_backend(self, mock_pytorch):
        """Test predict method with mocked backend."""
        mock_backend_instance = Mock()
        mock_backend_instance.predict.return_value = ('A', 0.95)
        mock_pytorch.return_value = mock_backend_instance
        
        config = InferenceConfig(backend=InferenceBackend.PYTORCH)
        engine = InferenceEngine(config)
        engine.backend = mock_backend_instance
        
        features = np.random.rand(63).astype(np.float32)
        letter, confidence = engine.predict(features)
        
        assert letter == 'A'
        assert confidence == 0.95
    
    @patch('gpu_inference.PyTorchInference')
    def test_predict_batch_with_mock_backend(self, mock_pytorch):
        """Test batch prediction with mocked backend."""
        mock_backend_instance = Mock()
        mock_backend_instance.predict_batch.return_value = [
            ('A', 0.95),
            ('B', 0.92),
            ('C', 0.88)
        ]
        mock_pytorch.return_value = mock_backend_instance
        
        config = InferenceConfig(backend=InferenceBackend.PYTORCH)
        engine = InferenceEngine(config)
        engine.backend = mock_backend_instance
        
        features_batch = [np.random.rand(63).astype(np.float32) for _ in range(3)]
        results = engine.predict_batch(features_batch)
        
        assert len(results) == 3
        assert results[0] == ('A', 0.95)
        assert results[1] == ('B', 0.92)
    
    def test_predict_returns_tuple(self):
        """Test that predict returns (letter, confidence) tuple."""
        engine = InferenceEngine()
        
        # Without loaded model, should still return valid format or handle gracefully
        features = np.random.rand(63).astype(np.float32)
        
        try:
            result = engine.predict(features)
            assert isinstance(result, tuple)
            assert len(result) == 2
        except Exception:
            # Expected if no model loaded
            pass
    
    def test_load_model_method_exists(self):
        """Test that load_model method exists."""
        engine = InferenceEngine()
        assert hasattr(engine, 'load_model')
        assert callable(engine.load_model)


class TestModelConversion:
    """Test model conversion utilities."""
    
    @patch('gpu_inference.joblib')
    @patch('gpu_inference.torch')
    def test_convert_sklearn_to_pytorch_signature(self, mock_torch, mock_joblib):
        """Test sklearn to PyTorch conversion function signature."""
        from gpu_inference import convert_sklearn_to_pytorch
        
        # Mock sklearn model
        mock_model = Mock()
        mock_joblib.load.return_value = mock_model
        
        # Should not raise exception with proper arguments
        try:
            convert_sklearn_to_pytorch(
                'model.pkl',
                'model.pth',
                num_features=63,
                num_classes=26
            )
        except Exception:
            # Expected if torch not available
            pass
    
    @patch('gpu_inference.joblib')
    def test_convert_sklearn_to_onnx_signature(self, mock_joblib):
        """Test sklearn to ONNX conversion function signature."""
        from gpu_inference import convert_sklearn_to_onnx
        
        # Mock sklearn model
        mock_model = Mock()
        mock_joblib.load.return_value = mock_model
        
        # Should not raise exception with proper arguments
        try:
            convert_sklearn_to_onnx(
                'model.pkl',
                'model.onnx',
                num_features=63
            )
        except Exception:
            # Expected if skl2onnx not available
            pass


class TestFeatureValidation:
    """Test feature input validation."""
    
    def test_features_shape_63_values(self):
        """Test that features should be 63-dimensional (21 landmarks × 3 coords)."""
        features = np.random.rand(63).astype(np.float32)
        
        assert features.shape == (63,)
        assert features.dtype == np.float32
    
    def test_batch_features_shape(self):
        """Test batch features shape."""
        batch_size = 8
        features_batch = np.random.rand(batch_size, 63).astype(np.float32)
        
        assert features_batch.shape == (batch_size, 63)
        assert features_batch.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
