import torch
from config import MODEL_CONFIGS, INFERENCE_BATCH_SIZE
from inference_pipeline import load_trained_model, evaluate_model, count_parameters, get_device

def run_inference(model_size: str = 'base'):
    """Run inference for a specific model size"""
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get config for model size
    config = MODEL_CONFIGS[model_size]
    model_id = config.hf_model_id
    print(f"Loading {model_size} model from HuggingFace: {model_id}")
    
    try:
        # Load model and dataset directly using config
        model, test_dataset = load_trained_model(config, device)
        
        # Count and display model parameters
        total_params, trainable_params = count_parameters(model)
        print("\nModel size:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("\nModel hyperparameters:")
        for key, value in config.__dict__.items():
            print(f"{key}: {value}")
        
        # Run evaluation
        print(f"\nRunning inference with batch size {INFERENCE_BATCH_SIZE}...")
        metrics = evaluate_model(
            model, 
            test_dataset, 
            batch_size=INFERENCE_BATCH_SIZE,
            device=device
        )
        
        # Print results
        print("\nResults:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        print("\nLatency Statistics:")
        print(f"Mean batch latency: {metrics['latency']['mean_batch_ms']:.2f} ms")
        print(f"Mean per-sample latency: {metrics['latency']['mean_per_sample_ms']:.2f} ms")
        print(f"P95 latency: {metrics['latency']['p95_ms']:.2f} ms")
        print(f"Min latency: {metrics['latency']['min_ms']:.2f} ms")
        print(f"Max latency: {metrics['latency']['max_ms']:.2f} ms")
        
        print("\nThroughput:")
        print(f"Samples per second: {metrics['throughput']['samples_per_second']:.2f}")
        print(f"Total samples: {metrics['throughput']['total_samples']:,}")
        print(f"Processed samples: {metrics['throughput']['processed_samples']:,}")
        print(f"Total inference time: {metrics['throughput']['total_time_seconds']:.2f} seconds")
        
    except Exception as e:
        print(f"Error running inference: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', choices=['small', 'base'], default='base',
                      help='Model size to run inference on')
    args = parser.parse_args()
    
    run_inference(args.model_size) 