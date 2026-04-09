import json
import os
import time
from datetime import datetime
from lolang.core.engine import Encoder, Decoder

class Benchmark:
    def __init__(self, model_names: list, seeds: list, dataset: list):
        self.model_names = model_names
        self.seeds = seeds
        self.dataset = dataset
        self.results = []

    def run(self):
        for model in self.model_names:
            for seed in self.seeds:
                encoder = Encoder(model=model, seed=seed)
                decoder = Decoder(model=model, seed=seed)
                
                for prompt in self.dataset:
                    start_time = time.time()
                    try:
                        encoded = encoder.encode(prompt)
                        decoded = decoder.decode(encoded)
                        elapsed = time.time() - start_time
                        
                        # Metrics
                        token_reduction = (1 - (len(encoded.split()) / len(prompt.split()))) * 100
                        accuracy = 1 if prompt.lower() in decoded.lower() or decoded.lower() in prompt.lower() else 0
                        
                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "model": model,
                            "seed": seed,
                            "original": prompt,
                            "encoded": encoded,
                            "decoded": decoded,
                            "token_reduction_pct": round(token_reduction, 2),
                            "accuracy": accuracy,
                            "latency_sec": round(elapsed, 2)
                        }
                        self.results.append(result)
                        print(f"[BENCHMARK] Model: {model} | Seed: {seed} | Reduction: {token_reduction}%")
                    except Exception as e:
                        print(f"[ERROR] {e}")
        
        self.save_logs()

    def save_logs(self):
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Benchmark results saved to {filename}")

if __name__ == "__main__":
    # Sample Test Set
    test_set = [
        "Do you have a convenient time to book a hotel room at 11pm?",
        "Can you help me summarize the latest AI trends in 2026?",
        "Tell me a joke about data science and machine learning."
    ]
    
    runner = Benchmark(
        model_names=["gemini-2.5-flash-lite", "gemini-3-flash-preview"],
        seeds=[279, 101, 555],
        dataset=test_set
    )
    runner.run()
