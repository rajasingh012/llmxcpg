import argparse
import json
import logging
import os
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
import random

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import jsonlines
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from unsloth import FastLanguageModel
import bitsandbytes as bnb
from transformers import PreTrainedTokenizer, PreTrainedModel

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    YES_TOKEN = "Yes"
    NO_TOKEN = "No"
    RANDOM_SEED = 42
    SYSTEM_PROMPT_PATH = './../prompts/detect_system_prompt.txt'
    BATCH_SIZE = 128
    OUTPUT_DIR = './results'
    LORA_CONFIG = {
        'r': 64,
        'lora_alpha': 128,
        'lora_dropout': 0,
        'bias': "none",
        'use_gradient_checkpointing': True
    }

class DataProcessor:
    """Handles data loading and preprocessing."""
    @staticmethod
    def distribute_cwes(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_cwes = {item['cwe'] for item in data if item['cwe'] not in ['N/A', 'NA', '']}
        valid_cwes = list(valid_cwes)
        
        cwe_counter = Counter(item['cwe'] for item in data if item['cwe'] not in ['N/A', 'NA', ''])
        logger.info(f"Original CWE distribution: {dict(cwe_counter)}")
        
        na_indices = [i for i, item in enumerate(data) if item['cwe'] in ['N/A', 'NA', '']]
        if na_indices:
            logger.info(f"Found {len(na_indices)} N/A CWEs to distribute")
            np.random.seed(1337)
            assigned_cwes = np.random.choice(valid_cwes, size=len(na_indices))
            for idx, cwe in zip(na_indices, assigned_cwes):
                data[idx]['cwe'] = cwe
        
        return data

    @staticmethod
    def load_data(dataset_path: str) -> List[Dict]:
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        with open(Config.SYSTEM_PROMPT_PATH, 'r') as f:
            system_prompt = f.read()
        
        processed_data = []
        for item in data:
            instruction = item['instruction'].replace("Single word: VULNERABLE or BENIGN", "Single word: Yes (if code is VULNERABLE) or No (if code is BENIGN)")

            label = 1 if item.get('output', item.get('gold', '')) in ["VULNERABLE", "Yes"] else 0

            dataset = ""
            if "formai" in dataset_path:
                dataset = "FormAI"
            elif "primevul" in dataset_path:
                dataset = "PrimeVul"
            elif "sven" in dataset_path:
                dataset = "SVEN"
            elif "reposvul" in dataset_path:
                dataset = "ReposVul"
            elif "realworld" in dataset_path:
                dataset = "RealWorld"
            entry = {
                "text": f"{system_prompt}\n\n## Instruction:\n{instruction}\n## Input:\n{item['input']}\n## Response:\n",
                "label": label,
                "dataset": dataset,
                "cwe": item.get('cwe', 'N/A'),
                "file_name": item.get('file_name', 'N/A'),
                "project": item.get('project', 'N/A'),
                "code": item.get('code', '')
            }
            processed_data.append(entry)

        random.shuffle(processed_data)
        return processed_data

class ModelHandler:
    """Handles model loading and setup."""
    def __init__(self, config: Config):
        self.config = config

    def find_linear_modules(self, model) -> List[str]:
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if 'lm_head' in lora_module_names:
                lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def load_model_and_tokenizer(self, model_path: str, base_model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        logger.info(f"Loading model and tokenizer from {model_path}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=16384,
        )
        
        yes_token_id = tokenizer.encode(self.config.YES_TOKEN, add_special_tokens=False)[0]
        no_token_id = tokenizer.encode(self.config.NO_TOKEN, add_special_tokens=False)[0]
        logger.info(f"Token IDs - Yes: {yes_token_id}, No: {no_token_id}")
        
        # Setup model head
        model.lm_head.weight = torch.nn.Parameter(torch.vstack([
            model.lm_head.weight[no_token_id, :],
            model.lm_head.weight[yes_token_id, :]
        ]))
        
        modules = self.find_linear_modules(model)
        logger.info(f"Loaded Modules: {modules}")
        
        # Setup LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            target_modules=modules,
            random_state=self.config.RANDOM_SEED,
            **self.config.LORA_CONFIG
        )
        
        model.load_adapter(model_path, adapter_name="default")
        FastLanguageModel.for_inference(model)
        model.eval()
        
        return model, tokenizer

class InferenceEngine:
    """Handles the inference process."""
    @staticmethod
    def prepare_batches(
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        batch_size: int
    ) -> Dict[int, List[Tuple]]:
        logger.info("Preparing batches for inference")
        tokenized_inputs = []
        
        for item in data:
            tokenized_input = tokenizer(item['text'], return_tensors="pt", add_special_tokens=False)
            tokenized_inputs.append((tokenized_input, item))
        
        tokenized_inputs.sort(key=lambda x: x[0]['input_ids'].shape[1])
        grouped_inputs = defaultdict(list)
        for item in tokenized_inputs:
            length = item[0]['input_ids'].shape[1]
            grouped_inputs[length].append(item)
        
        return grouped_inputs

    @staticmethod
    def run_inference(
        model: PreTrainedModel,
        grouped_inputs: Dict[int, List[Tuple]],
        batch_size: int
    ) -> List[Dict]:
        logger.info("Running inference")
        all_results = []
        prediction_counts = defaultdict(int)
        confidence_stats = []
        
        logger.info(f"Model head shape: {model.lm_head.weight.shape}")
        
        for length, group in tqdm(grouped_inputs.items(), desc="Processing batches"):
            for i in range(0, len(group), batch_size):
                batch = group[i:i+batch_size]
                batch_inputs = [item[0] for item in batch]
                batch_items = [item[1] for item in batch]
                
                input_ids = torch.cat([item['input_ids'] for item in batch_inputs], dim=0).to("cuda")
                attention_mask = torch.cat([item['attention_mask'] for item in batch_inputs], dim=0).to("cuda")
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                last_token_logits = outputs.logits[:, -1]
                probabilities = F.softmax(last_token_logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                for pred, item, prob, logits_row in zip(predictions, batch_items, probabilities, last_token_logits):
                    pred_int = int(pred.cpu())
                    confidence = float(prob[pred_int].cpu())
                    prediction_counts[pred_int] += 1
                    confidence_stats.append(confidence)
                    
                    result = {
                        'code': item['text'].split("\n## Input:\n")[-1].replace("\n## Response:\n", ""),
                        'prediction': pred_int,
                        'label': item['label'],
                        'dataset': item['dataset'],
                        'cwe': item['cwe'],
                        'file_name': item['file_name'],
                        'project': item['project'],
                        'confidence': confidence,
                        'probabilities': prob.cpu().tolist(),
                        'probability_vulnerable': float(prob[1].cpu())
                    }
                    all_results.append(result)
        
        InferenceEngine._log_statistics(prediction_counts, confidence_stats)
        return all_results

    @staticmethod
    def _log_statistics(prediction_counts: Dict[int, int], confidence_stats: List[float]):
        logger.info("Prediction distribution:")
        total = sum(prediction_counts.values())
        for pred, count in prediction_counts.items():
            label_name = "No/Benign" if pred == 0 else "Yes/Vulnerable"
            logger.info(f"Class {pred} ({label_name}): {count} ({count/total*100:.2f}%)")
        
        logger.info("Confidence statistics:")
        confidence_array = np.array(confidence_stats)
        logger.info(f"Mean confidence: {np.mean(confidence_array):.4f}")
        logger.info(f"Std confidence: {np.std(confidence_array):.4f}")
        logger.info(f"Min confidence: {np.min(confidence_array):.4f}")
        logger.info(f"Max confidence: {np.max(confidence_array):.4f}")

class MetricsCalculator:
    """Calculates and evaluates metrics."""
    @staticmethod
    def compute_metrics(results: List[Dict], threshold: float) -> Dict:
        predictions = [1 if r['probability_vulnerable'] >= threshold else 0 for r in results]
        labels = [r['label'] for r in results]
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        tp = sum((p == 1 and l == 1) for p, l in zip(predictions, labels))
        fp = sum((p == 1 and l == 0) for p, l in zip(predictions, labels))
        fn = sum((p == 0 and l == 1) for p, l in zip(predictions, labels))
        tn = sum((p == 0 and l == 0) for p, l in zip(predictions, labels))
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {
                'tp': tp, 'fp': fp,
                'fn': fn, 'tn': tn
            },
            'total_samples': len(predictions)
        }

class ResultsHandler:
    """Handles saving and logging results."""
    @staticmethod
    def save_results(results: List[Dict], metrics: Dict, base_model: str, dataset_path: str, output_dir: str):
        model_name = base_model.split('/')[-1]
        dataset_name = dataset_path.split("/")[-1].replace(".json", "")
        base_filename = f"{model_name}_{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        predictions_file = os.path.join(output_dir, f'{base_filename}_predictions.jsonl')
        with jsonlines.open(predictions_file, mode='w') as writer:
            for result in results:
                writer.write({
                    'code': result['code'],
                    'prediction': 1 if result['probability_vulnerable'] >= metrics['threshold'] else 0,
                    'label': result['label'],
                    'probability_vulnerable': result['probability_vulnerable']
                })
        
        # Save metrics to TSV
        metrics_file = os.path.join(output_dir, f'{base_filename}_metrics.tsv')
        with open(metrics_file, 'w') as f:
            f.write(f"model\tdataset\tthreshold\taccuracy\tprecision\trecall\tf1\ttotal_samples\n")
            f.write(f"{model_name}\t{dataset_name}\t{metrics['threshold']}\t{metrics['accuracy']}\t{metrics['precision']}\t{metrics['recall']}\t{metrics['f1']}\t{metrics['total_samples']}\n")

    @staticmethod
    def log_results(model: str, results: dict, dataset: str):
        with open("final_inference_results.tsv", "a") as f:
            f.write(f"{model}\t{dataset}\t{results['threshold']}\t{results['accuracy']}\t{results['precision']}\t{results['recall']}\t{results['f1']}\t{results['total_samples']}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Run model inference on dataset")
    parser.add_argument('dataset', choices=['primevul', 'formai', 'reposvul', 'pkco', 'sven'],
                        help="Dataset to evaluate: primevul, formai, reposvul, pkco, or sven")
    parser.add_argument('--base-model', type=str, default="/workspace/QwQ-32B-Preview", help="Get the model from Qwen/QwQ-32B-Preview on HF")
    parser.add_argument('--model-path', type=str, default="/workspace/QCRI__LLMxCPG-D", help="Get the model from QCRI/LLMxCPG-D on HF")
    parser.add_argument('--threshold', type=float, default=None, help="Classification threshold (default depends on dataset)")
    parser.add_argument('--output-dir', default=Config.OUTPUT_DIR, help="Directory to save output files")
    
    args = parser.parse_args()

    dataset_files = {
        'primevul': './../data/primevul_test.json',
        'formai': './../data/formai_test.json',
        'reposvul': './../data/reposvul_test.json',
        'pkco': './../data/pkco_test.json',
        'sven': './../data/sven_test.json'
    }

    default_thresholds = {
        'formai': 0.547,
        'reposvul': 0.307,
        'primevul': 0.594,
        'sven': 0.334,
        'pkco': 0.32
    }

    args.dataset_path = dataset_files[args.dataset]
    if args.threshold is None:
        args.threshold = default_thresholds[args.dataset]

    return args


def main():
    """Main execution function."""
    try:
        args = parse_args()
        config = Config()
        
        # Initialize components
        model_handler = ModelHandler(config)
        model, tokenizer = model_handler.load_model_and_tokenizer(args.model_path, args.base_model)
        
        # Load and process data
        data = DataProcessor.load_data(args.dataset_path)
        
        # Prepare for inference
        grouped_inputs = InferenceEngine.prepare_batches(data, tokenizer, config.BATCH_SIZE)
        results = InferenceEngine.run_inference(model, grouped_inputs, config.BATCH_SIZE)
        
        # Calculate metrics with given threshold
        metrics = MetricsCalculator.compute_metrics(results, threshold=args.threshold)
        
        logger.info("\nMetrics with threshold {:.2f}:".format(args.threshold))
        for k, v in metrics.items():
            if isinstance(v, dict):
                logger.info(f"{k}:")
                for sk, sv in v.items():
                    logger.info(f"  {sk}: {sv}")
            else:
                logger.info(f"{k}: {v}")
        
        # Save results
        ResultsHandler.save_results(results, metrics, args.base_model, args.dataset_path, args.output_dir)
        ResultsHandler.log_results(
            model=args.base_model,
            results=metrics,
            dataset=args.dataset_path.split("/")[-1].replace(".json", "")
        )

    except Exception as e:
        logger.exception("An error occurred during inference")
        raise

if __name__ == "__main__":
    main()
