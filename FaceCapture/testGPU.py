import tensorflow as tf
import sys
import subprocess

print("=" * 50)
print("SYSTEM INFORMATION")
print("=" * 50)

# Check Python and TensorFlow
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

# Check GPU availability
print(f"\nGPU CHECK:")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
#print(f"Built with cuDNN: {tf.test.is_built_with_cudnn()}")

# List all GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nFound {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        
    # Enable memory growth to avoid OOM
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("  ✓ Memory growth enabled")
    except RuntimeError as e:
        print(f"  ✗ Error setting memory growth: {e}")
else:
    print("\n✗ No GPU found!")

# Test GPU performance
print("\n" + "=" * 50)
print("PERFORMANCE TEST")
print("=" * 50)

if gpus:
    print("Testing GPU performance with a simple matrix multiplication...")
    with tf.device('/GPU:0'):
        # Create large tensors
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        
        # Time the operation
        import time
        start = time.time()
        c = tf.matmul(a, b)
        end = time.time()
        
        print(f"  Matrix multiplication time: {end - start:.2f} seconds")
        print(f"  Result shape: {c.shape}")
        print("  ✓ GPU is working!")
else:
    print("Skipping performance test (no GPU)")

# Test DeepFace GPU usage
print("\n" + "=" * 50)
print("DEEPFACE GPU TEST")
print("=" * 50)

try:
    from deepface import DeepFace
    import numpy as np
    
    print("Building DeepFace models with GPU...")
    
    # Test with a small model first
    model_names = ['Facenet512']
    
    for model_name in model_names:
        try:
            print(f"\nLoading {model_name}...")
            model = DeepFace.build_model(model_name)
            print(f"  ✓ {model_name} loaded successfully")
            
            # Check if model is on GPU
            if hasattr(model, 'layers'):
                first_layer = model.layers[0] if model.layers else None
                if first_layer and hasattr(first_layer, 'weights'):
                    weights = first_layer.weights
                    if weights:
                        device = weights[0].device
                        print(f"  Model weights on: {device}")
        except Exception as e:
            print(f"  ✗ Error loading {model_name}: {e}")
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if gpus:
        print("✅ SUCCESS: TensorFlow is using your RTX 4080 GPU!")
        print("\nFor optimal DeepFace performance with RTX 4080:")
        print("1. Use batch processing when analyzing multiple faces")
        print("2. Consider using larger models (Facenet512, ArcFace)")
        print("3. Monitor GPU usage: watch -n 1 nvidia-smi")
    else:
        print("❌ FAILED: TensorFlow is NOT using GPU")
        print("\nTroubleshooting steps:")
        print("1. Verify CUDA/cuDNN installation: nvcc --version")
        print("2. Check driver: nvidia-smi")
        print("3. Reinstall TensorFlow-GPU: pip install tensorflow-gpu==2.10.0")
        
except ImportError as e:
    print(f"Error importing DeepFace: {e}")