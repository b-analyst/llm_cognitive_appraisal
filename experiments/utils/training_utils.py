"""
Training utilities for binary OVA (One-vs-All) emotion probes.

This module contains the OvaLogisticRegressionTrainer class for training
layerwise binary probes for emotion classification.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score
)
from tqdm import tqdm
from joblib import Parallel, delayed
from transformers import AutoTokenizer, AutoModelForCausalLM


def _probe_suffix_safe(layers, locs, tokens, model_name=None):
    """
    Filesystem-safe suffix for .pt filenames (no brackets/commas for Windows).

    Use a compact encoding to avoid very long paths on Windows, especially for
    deeper models where layers may be 0..41, 0..63, etc.
    """
    if len(layers) == 0:
        layer_part = "layers_unknown"
    else:
        layer_part = f"layers_{layers[0]}-{layers[-1]}"
    loc_part = f"locs_{'_'.join(str(x) for x in locs)}" if locs else "locs_unknown"
    tok_part = f"tokens_{'_'.join(str(t) for t in tokens)}" if tokens else "tokens_unknown"
    if model_name:
        safe_model = "".join(ch if ch.isalnum() else "_" for ch in str(model_name))
        return f"{safe_model}_{layer_part}_{loc_part}_{tok_part}"
    return f"{layer_part}_{loc_part}_{tok_part}"


# Import model classes
if '..' not in sys.path:
    sys.path.append('..')

try:
    from LLMs.my_llama import LlamaForCausalLM
    from LLMs.my_phi3 import Phi3ForCausalLM
    from LLMs.my_gemma2 import Gemma2ForCausalLM
    from LLMs.my_olmo import OlmoForCausalLM
    from LLMs.my_mistral import MistralForCausalLM
    from LLMs.my_olmo2 import Olmo2ForCausalLM
except Exception:
    LlamaForCausalLM = None
    Phi3ForCausalLM = None
    Gemma2ForCausalLM = None
    OlmoForCausalLM = None
    MistralForCausalLM = None
    Olmo2ForCausalLM = None


class OvaLogisticRegressionTrainer:
    """
    Trainer class for binary OVA (One-vs-All) emotion probes.
    
    Handles loading datasets, extracting hidden states, training probes, and saving results.
    
    Example usage:
        ```python
        from experiments.utils.training_utils import OvaLogisticRegressionTrainer
        from utils import Log
        
        trainer = OvaLogisticRegressionTrainer('Llama3.2_1B', logger=Log('training').logger)
        results = trainer.train(
            datasets_dir='outputs/binary_ova_datasets',
            emotions_list=['anger', 'boredom', 'disgust', 'fear', 'guilt', 'joy', 
                          'pride', 'relief', 'sadness', 'shame', 'surprise', 'trust'],
            output_dir='outputs/binary_ova_probes',
            extraction_layers=None,  # Uses all layers
            extraction_locs=[3, 6, 7],
            extraction_tokens=[-1]
        )
        ```
    """
    
    # Model mappings
    AVAILABLE_HF_MODELS = {
        'Llama3.2_1B': 'meta-llama/Llama-3.2-1B-Instruct',
        'Llama3.1_8B': 'meta-llama/Llama-3.1-8B-Instruct',
        'Gemma2_2B': 'google/gemma-2-2b-it',
        'Gemma2_9B': 'google/gemma-2-9b-it',
        'Phi3_4B': 'microsoft/Phi-3.5-mini-instruct',
        'Phi3_14B': 'microsoft/Phi-3-medium-128k-instruct',
        'Mistral_8B': 'mistralai/Ministral-8B-Instruct-2410',
        'Mistral_12B': 'mistralai/Mistral-Nemo-Instruct-2407',
        'OLMo2_7B': 'allenai/OLMo-2-1124-7B-Instruct',
        'OLMo2_13B': 'allenai/OLMo-2-1124-13B-Instruct',
    }
    
    MODEL_CLASSES = {
        'Llama3.2_1B': LlamaForCausalLM,
        'Llama3.1_8B': LlamaForCausalLM,
        'Gemma2_2B': Gemma2ForCausalLM,
        'Gemma2_9B': Gemma2ForCausalLM,
        'Phi3_4B': Phi3ForCausalLM,
        'Phi3_14B': Phi3ForCausalLM,
        'Mistral_8B': MistralForCausalLM,
        'Mistral_12B': MistralForCausalLM,
        'OLMo2_7B': Olmo2ForCausalLM,
        'OLMo2_13B': Olmo2ForCausalLM,
    }

    USE_LEGACY_CUSTOM_MODELS = os.environ.get("PIPELINE_USE_LEGACY_MODELS", "0") == "1"
    
    def __init__(self, model_name, device_map='auto', logger=None):
        """
        Initialize the trainer.
        
        Args:
            model_name: Short name (e.g., 'Llama3.2_1B') or full HuggingFace path
            device_map: Device mapping for model loading
            logger: Optional logger instance
        """
        self.device_map = device_map
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.num_layers = None
        
        # Determine model info
        if model_name in self.AVAILABLE_HF_MODELS:
            self.model_short_name = model_name
            self.hf_model_name = self.AVAILABLE_HF_MODELS[model_name]
            self.model_class = self.MODEL_CLASSES[model_name]
        else:
            # Assume it's a full HF path - try to infer model class
            self.hf_model_name = model_name
            self.model_short_name = None
            self.model_class = self._infer_model_class(model_name)
    
    def _infer_model_class(self, hf_model_name):
        """Infer model class from HuggingFace model name."""
        name_lower = hf_model_name.lower()
        if 'llama' in name_lower:
            return LlamaForCausalLM
        elif 'gemma' in name_lower:
            return Gemma2ForCausalLM
        elif 'phi' in name_lower:
            return Phi3ForCausalLM
        elif 'mistral' in name_lower or 'ministral' in name_lower:
            return MistralForCausalLM
        elif 'olmo' in name_lower:
            if 'olmo-2' in name_lower or 'olmo2' in name_lower:
                return Olmo2ForCausalLM
            return OlmoForCausalLM
        else:
            raise ValueError(f"Could not infer model class for {hf_model_name}")

    @staticmethod
    def _tensor_to_numpy_safe(x):
        """Convert tensor-like hidden states to NumPy, normalizing bf16/fp16 to float32 first."""
        if isinstance(x, torch.Tensor):
            t = x.detach()
            if t.is_floating_point() and t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            return t.cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _cache_hidden_state_block(x):
        """Store reusable hidden states in compact CPU format."""
        arr = OvaLogisticRegressionTrainer._tensor_to_numpy_safe(x)
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float16, copy=False)
        return arr

    def _materialize_hidden_states_from_cache(
        self,
        texts,
        split_name,
        cache_store,
        extraction_layers,
        extraction_locs,
        extraction_tokens,
        batch_size,
    ):
        """Deduplicate inference by extracting only unseen texts for this split."""
        ordered_texts = [str(t) for t in texts]
        missing = []
        seen_missing = set()
        for text in ordered_texts:
            if text not in cache_store and text not in seen_missing:
                seen_missing.add(text)
                missing.append(text)
        if missing:
            if self.logger:
                self.logger.info(
                    f"{split_name}: extracting {len(missing)} unseen texts "
                    f"(cache hits so far: {len(ordered_texts) - len(missing)})"
                )
            missing_hs = self.extract_hidden_states(
                texts=missing,
                extraction_layers=extraction_layers,
                extraction_locs=extraction_locs,
                extraction_tokens=extraction_tokens,
                batch_size=batch_size,
            )
            missing_np = self._cache_hidden_state_block(missing_hs)
            for idx, text in enumerate(missing):
                cache_store[text] = missing_np[idx]
            del missing_hs
            del missing_np
        return np.stack([cache_store[text] for text in ordered_texts], axis=0)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer
        
        if self.logger:
            self.logger.info(f"Loading model: {self.hf_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {}
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                load_kwargs["torch_dtype"] = torch.bfloat16
            else:
                load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["attn_implementation"] = "sdpa"
            # RTX 4090-class GPUs benefit from TF32 kernels for matmuls where fp32 is used.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        else:
            load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["low_cpu_mem_usage"] = True

        model_cls = self.model_class if (self.USE_LEGACY_CUSTOM_MODELS and self.model_class is not None) else AutoModelForCausalLM
        if self.logger and model_cls is AutoModelForCausalLM and self.model_class is not None:
            self.logger.info("Using stock AutoModelForCausalLM hook backend (legacy custom wrappers disabled by default).")
        elif self.logger and model_cls is AutoModelForCausalLM:
            self.logger.info("Using stock AutoModelForCausalLM hook backend.")
        if self.logger:
            self.logger.info(
                f"Model load kwargs: device_map={load_kwargs.get('device_map')}, "
                f"dtype={load_kwargs.get('torch_dtype')}, "
                f"attn_impl={load_kwargs.get('attn_implementation', 'default')}"
            )

        self.model = model_cls.from_pretrained(
            self.hf_model_name,
            **load_kwargs,
        )
        self.model.eval()
        if getattr(self.model, "generation_config", None) is not None and self.tokenizer.pad_token_id is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        # Get number of layers
        if hasattr(self.model.config, 'num_hidden_layers'):
            self.num_layers = self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'num_layers'):
            self.num_layers = self.model.config.num_layers
        else:
            self.num_layers = None
        
        if self.logger:
            self.logger.info(f"Model loaded. Layers: {self.num_layers}")
        
        return self.model, self.tokenizer
    
    def extract_hidden_states(self, texts, extraction_layers, extraction_locs, 
                             extraction_tokens, batch_size=1, logger=None):
        """
        Extract hidden states from model.
        
        Args:
            texts: List of texts
            extraction_layers: List of layer indices
            extraction_locs: List of location indices (3, 6, 7, etc.)
            extraction_tokens: List of token indices (e.g., [-1] for last token)
            batch_size: Batch size
            logger: Optional logger
        
        Returns:
            Tensor of shape (n_samples, n_layers, n_locs, n_tokens, n_features)
        """
        from utils import extract_hidden_states as extract_hs, TextDataset
        from torch.utils.data import DataLoader
        
        if logger is None:
            logger = self.logger
        
        # Load model if needed
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Convert texts to list if needed
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = list(texts)
        
        # Create dummy labels (not used by extract_hidden_states)
        dummy_labels = [0] * len(texts)
        
        # Create dataset and dataloader
        dataset = TextDataset(texts=texts, labels=dummy_labels)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        
        hidden_states = extract_hs(
            dataloader=dataloader,
            tokenizer=self.tokenizer,
            model=self.model,
            logger=logger,
            extraction_layers=extraction_layers,
            extraction_locs=extraction_locs,
            extraction_tokens=extraction_tokens
        )
        
        return hidden_states
    
    def _load_binary_ova_datasets(self, datasets_dir, emotions_list, text_column='hidden_emo_text'):
        """Load binary OVA datasets using BinaryOvaDatasetProcessor."""
        from experiments.utils.data_utils import BinaryOvaDatasetProcessor
        
        processor = BinaryOvaDatasetProcessor(
            output_dir=datasets_dir,
            emotions_list=emotions_list,
            logger=self.logger
        )
        
        return processor.load_datasets(text_column=text_column)
    
    def _train_binary_probe_simple(self, X_train, y_train, X_val=None, y_val=None,
                                   C=1.0, C_grid=None, normalize=True, random_state=42, max_iter=5000):
        """Train a single binary probe."""
        # Use validation data if provided, otherwise split training data
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
            )
        def _safe_roc_auc(y_true, y_score):
            y_true = np.asarray(y_true)
            if np.unique(y_true).size < 2:
                return float("nan")
            return float(roc_auc_score(y_true, y_score))

        def _safe_pr_auc(y_true, y_score):
            y_true = np.asarray(y_true)
            if np.unique(y_true).size < 2:
                return float("nan")
            return float(average_precision_score(y_true, y_score))

        candidate_cs = list(C_grid) if C_grid is not None else [C]
        best = None
        best_score = (-1.0, -1.0)
        for idx, c_val in enumerate(candidate_cs):
            scaler = None
            X_train_fit = X_train
            X_val_fit = X_val
            if normalize:
                scaler = StandardScaler()
                X_train_fit = scaler.fit_transform(X_train)
                X_val_fit = scaler.transform(X_val)

            clf = LogisticRegression(
                C=float(c_val),
                solver='lbfgs',
                class_weight='balanced',
                max_iter=max_iter,
                random_state=random_state + idx,
                tol=1e-3
            )
            clf.fit(X_train_fit, y_train)
            y_val_prob = clf.predict_proba(X_val_fit)[:, 1]
            y_val_pred = clf.predict(X_val_fit)
            val_roc = _safe_roc_auc(y_val, y_val_prob)
            val_acc = float(accuracy_score(y_val, y_val_pred))
            score_tuple = ((-1.0 if np.isnan(val_roc) else val_roc), val_acc)
            if score_tuple > best_score:
                best_score = score_tuple
                best = (clf, scaler, X_train_fit, X_val_fit, y_val_prob, y_val_pred, float(c_val))

        clf, scaler, X_train_fit, X_val_fit, y_val_prob, y_val_pred, best_c = best
        y_train_pred = clf.predict(X_train_fit)
        y_train_prob = clf.predict_proba(X_train_fit)[:, 1]

        if self.logger and np.unique(y_val).size < 2:
            self.logger.warning(
                "Validation split has only one class; ROC-AUC/PR-AUC will be recorded as NaN for this probe."
            )

        metrics = {
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'train_f1': float(f1_score(y_train, y_train_pred, zero_division=0)),
            'train_roc_auc': _safe_roc_auc(y_train, y_train_prob),
            'train_pr_auc': _safe_pr_auc(y_train, y_train_prob),
            'test_accuracy': float(accuracy_score(y_val, y_val_pred)),
            'test_f1': float(f1_score(y_val, y_val_pred, zero_division=0)),
            'test_roc_auc': _safe_roc_auc(y_val, y_val_prob),
            'test_pr_auc': _safe_pr_auc(y_val, y_val_prob),
            'selected_C': float(best_c),
        }
        
        return {
            'classifier': clf,
            'scaler': scaler,
            'metrics': metrics,
            'weights': clf.coef_.ravel(),
            'bias': clf.intercept_[0],
            'X_test': X_val,
            'y_test': y_val,
            'y_test_prob': y_val_prob,
            'y_test_pred': y_val_pred,
            'selected_C': float(best_c),
        }
    
    def _train_layerwise_probes_single_emotion(self, hidden_states, labels, emotion,
                                               extraction_layers, extraction_locs, extraction_tokens,
                                               val_hidden_states=None, val_labels=None,
                                               C=1.0, C_grid=None, normalize=True, random_state=42, max_iter=5000,
                                               n_jobs_probes=1):
        """Train binary OVA probes for a single emotion across all layers/locs/tokens."""
        results = {layer: {loc: {} for loc in extraction_locs} for layer in extraction_layers}
        
        # Determine token indices
        if isinstance(extraction_tokens, str) and extraction_tokens == 'auto':
            token_indices = range(hidden_states.shape[3])
            token_keys = list(token_indices)
        else:
            token_indices = range(len(extraction_tokens))
            token_keys = extraction_tokens
        
        # Convert labels to numpy
        if isinstance(labels, torch.Tensor):
            y = labels.cpu().numpy()
        else:
            y = np.asarray(labels).ravel()
        
        # Convert validation labels to numpy if provided
        y_val = None
        if val_labels is not None:
            if isinstance(val_labels, torch.Tensor):
                y_val = val_labels.cpu().numpy()
            else:
                y_val = np.asarray(val_labels).ravel()
        
        # Check if we have enough samples
        pos_count = int(y.sum())
        neg_count = int(len(y) - pos_count)
        
        if min(pos_count, neg_count) < 2:
            if self.logger:
                self.logger.warning(f"{emotion}: Skipping - insufficient samples (pos={pos_count}, neg={neg_count})")
            return results
        
        n_probes = len(extraction_layers) * len(extraction_locs) * len(token_keys)
        val_info = f"with validation set" if val_hidden_states is not None else "with train/test split"
        if self.logger:
            self.logger.info(f"{emotion}: Training {n_probes} probes ({len(extraction_layers)} layers × {len(extraction_locs)} locs × {len(token_keys)} tokens) ({pos_count} pos, {neg_count} neg) {val_info}")
        
        hs_np = self._tensor_to_numpy_safe(hidden_states)
        val_hs_np = None
        if val_hidden_states is not None:
            val_hs_np = self._tensor_to_numpy_safe(val_hidden_states)

        tasks = [
            (layer_idx, layer, loc_idx, loc, token_idx, token_key)
            for layer_idx, layer in enumerate(extraction_layers)
            for loc_idx, loc in enumerate(extraction_locs)
            for token_idx, token_key in enumerate(token_keys)
        ]

        def _fit_one(task):
            layer_idx, layer, loc_idx, loc, token_idx, token_key = task
            X_train = hs_np[:, layer_idx, loc_idx, token_idx, :].reshape(hs_np.shape[0], -1)
            if X_train.shape[0] != len(y):
                return layer, loc, token_key, {'error': f'Dimension mismatch: X_train has {X_train.shape[0]} samples, y has {len(y)} labels'}
            X_val = None
            if val_hs_np is not None:
                X_val = val_hs_np[:, layer_idx, loc_idx, token_idx, :].reshape(val_hs_np.shape[0], -1)
                if X_val.shape[0] != len(y_val):
                    return layer, loc, token_key, {'error': f'Dimension mismatch: X_val has {X_val.shape[0]} samples, y_val has {len(y_val)} labels'}
            try:
                probe_result = self._train_binary_probe_simple(
                    X_train, y, X_val=X_val, y_val=y_val,
                    C=C, C_grid=C_grid, normalize=normalize,
                    random_state=random_state, max_iter=max_iter
                )
                return layer, loc, token_key, probe_result
            except Exception as e:
                return layer, loc, token_key, {'error': str(e)}

        n_jobs = int(n_jobs_probes) if n_jobs_probes is not None else 1
        if n_jobs <= 1:
            fitted = [_fit_one(t) for t in tasks]
        else:
            fitted = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_fit_one)(t) for t in tasks)

        for layer, loc, token_key, probe_result in fitted:
            results[layer][loc][token_key] = probe_result
        
        return results
    
    def _summarize_probe_results(self, results, emotions_list, extraction_layers, extraction_locs, extraction_tokens):
        """Create a summary DataFrame of probe results."""
        summary_rows = []
        
        for layer in extraction_layers:
            for loc in extraction_locs:
                for token in extraction_tokens:
                    for emotion in emotions_list:
                        if layer not in results:
                            continue
                        if loc not in results[layer]:
                            continue
                        if token not in results[layer][loc]:
                            continue
                        if emotion not in results[layer][loc][token]:
                            continue
                        
                        result = results[layer][loc][token][emotion]
                        
                        if 'error' in result:
                            continue
                        
                        # Compute mean confidence score
                        mean_confidence = None
                        if 'y_test_prob' in result:
                            mean_confidence = float(np.mean(result['y_test_prob']))
                        
                        row = {
                            'layer': layer,
                            'loc': loc,
                            'token': token,
                            'emotion': emotion,
                            'selected_C': result.get('selected_C'),
                            'test_accuracy': result['metrics']['test_accuracy'],
                            'test_f1': result['metrics']['test_f1'],
                            'test_roc_auc': result['metrics']['test_roc_auc'],
                            'test_pr_auc': result['metrics']['test_pr_auc'],
                            'train_accuracy': result['metrics']['train_accuracy'],
                            'train_f1': result['metrics']['train_f1'],
                        }
                        
                        if mean_confidence is not None:
                            row['mean_confidence'] = mean_confidence
                        
                        summary_rows.append(row)
        
        if len(summary_rows) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(summary_rows)
    
    def train(self, datasets_dir, emotions_list, output_dir,
              extraction_layers=None, extraction_locs=[3, 6, 7], extraction_tokens=[-1],
              C=1.0, C_grid=None, normalize=True, random_state=42, max_iter=5000,
              batch_size=1, text_column='hidden_emo_text', 
              save_hidden_states=True, save_by_location=True, n_jobs_probes=1):
        """
        Main training method.
        
        Loads datasets, extracts hidden states, trains probes, and saves results.
        
        Args:
            datasets_dir: Directory containing processed datasets (from BinaryOvaDatasetProcessor)
            emotions_list: List of emotion names
            output_dir: Directory to save results
            extraction_layers: List of layer indices (None = all layers)
            extraction_locs: List of location indices
            extraction_tokens: List of token indices
            C: Regularization strength
            normalize: Whether to normalize features
            random_state: Random seed
            max_iter: Maximum iterations
            batch_size: Batch size for hidden state extraction
            text_column: Name of text column in datasets
            save_hidden_states: Whether to save extracted hidden states
            save_by_location: Whether to save results separately by location
        
        Returns:
            Dictionary with 'results', 'summary_df', and 'output_dir'
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.logger:
            self.logger.info("Starting binary OVA probe training...")
            self.logger.info(f"Processing {len(emotions_list)} emotions sequentially")
        
        # Load model and tokenizer
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Get extraction layers
        if extraction_layers is None:
            if self.num_layers is None:
                raise ValueError("Number of layers unknown. Please specify extraction_layers manually.")
            extraction_layers = list(range(self.num_layers))
        
        if self.logger:
            self.logger.info(f"Extracting from layers: {extraction_layers}")
            self.logger.info(f"Extracting from locations: {extraction_locs}")
            self.logger.info(f"Extracting from tokens: {extraction_tokens}")
        
        # Load datasets
        if self.logger:
            self.logger.info(f"Loading datasets from: {datasets_dir}")
        train_labels_dict, train_texts_dict, val_labels_dict, val_texts_dict = self._load_binary_ova_datasets(
            datasets_dir, emotions_list, text_column
        )
        
        if len(train_labels_dict) == 0:
            raise ValueError(f"No training datasets loaded from {datasets_dir}")
        
        if self.logger:
            self.logger.info(f"Loaded training datasets for {len(train_labels_dict)} emotions")
            if len(val_labels_dict) > 0:
                self.logger.info(f"Loaded validation datasets for {len(val_labels_dict)} emotions")
        
        # Initialize results structure
        all_results = {}
        train_text_cache = {}
        val_text_cache = {}
        
        # Process each emotion separately to save memory
        for emotion in tqdm(emotions_list, desc="Emotions"):
            if emotion not in train_labels_dict or emotion not in train_texts_dict:
                if self.logger:
                    self.logger.warning(f"Skipping {emotion}: training dataset not found")
                continue
            
            # Get texts and labels
            texts = train_texts_dict[emotion]
            labels = train_labels_dict[emotion]
            val_texts = val_texts_dict.get(emotion)
            val_labels = val_labels_dict.get(emotion)
            
            # Convert numpy arrays to lists if needed
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            elif not isinstance(texts, list):
                texts = list(texts)
            
            if isinstance(labels, np.ndarray):
                labels = labels
            elif not isinstance(labels, np.ndarray):
                labels = np.asarray(labels)
            
            if val_texts is not None:
                if isinstance(val_texts, np.ndarray):
                    val_texts = val_texts.tolist()
                elif not isinstance(val_texts, list):
                    val_texts = list(val_texts)
            
            if val_labels is not None:
                if isinstance(val_labels, np.ndarray):
                    val_labels = val_labels
                elif not isinstance(val_labels, np.ndarray):
                    val_labels = np.asarray(val_labels)
            
            if self.logger:
                self.logger.info(f"Processing {emotion} ({len(texts)} train samples, {len(val_texts) if val_texts is not None else 0} val samples)")
            
            # Check if hidden states already exist (use filesystem-safe suffix for Windows)
            _suffix = _probe_suffix_safe(
                extraction_layers,
                extraction_locs,
                extraction_tokens,
                model_name=self.model_short_name or self.hf_model_name,
            )
            hidden_states_path = os.path.join(
                output_dir, 
                f'hidden_states_{emotion}_{_suffix}.pt'
            )
            
            if os.path.exists(hidden_states_path):
                if self.logger:
                    self.logger.info(f"Loading existing hidden states for {emotion}")
                hidden_states = torch.load(hidden_states_path, weights_only=False)
            else:
                hidden_states = self._materialize_hidden_states_from_cache(
                    texts=texts,
                    split_name=f"{emotion}/train",
                    cache_store=train_text_cache,
                    extraction_layers=extraction_layers,
                    extraction_locs=extraction_locs,
                    extraction_tokens=extraction_tokens,
                    batch_size=batch_size
                )
                
                # Save hidden states if requested
                if save_hidden_states:
                    torch.save(hidden_states, hidden_states_path)
                    if self.logger:
                        self.logger.info(f"Saved hidden states for {emotion}")
            
            # Extract validation hidden states if available
            val_hidden_states = None
            if val_texts is not None and val_labels is not None:
                val_hidden_states_path = os.path.join(
                    output_dir, 
                    f'hidden_states_{emotion}_val_{_suffix}.pt'
                )
                
                if os.path.exists(val_hidden_states_path):
                    if self.logger:
                        self.logger.info(f"Loading existing validation hidden states for {emotion}")
                    val_hidden_states = torch.load(val_hidden_states_path, weights_only=False)
                else:
                    val_hidden_states = self._materialize_hidden_states_from_cache(
                        texts=val_texts,
                        split_name=f"{emotion}/selection",
                        cache_store=val_text_cache,
                        extraction_layers=extraction_layers,
                        extraction_locs=extraction_locs,
                        extraction_tokens=extraction_tokens,
                        batch_size=batch_size
                    )
                    
                    if save_hidden_states:
                        torch.save(val_hidden_states, val_hidden_states_path)
            
            # Train probes
            emotion_results = self._train_layerwise_probes_single_emotion(
                hidden_states=hidden_states,
                labels=labels,
                emotion=emotion,
                extraction_layers=extraction_layers,
                extraction_locs=extraction_locs,
                extraction_tokens=extraction_tokens,
                val_hidden_states=val_hidden_states,
                val_labels=val_labels,
                C=C,
                C_grid=C_grid,
                normalize=normalize,
                random_state=random_state,
                max_iter=max_iter,
                n_jobs_probes=n_jobs_probes,
            )
            
            # Store results
            for layer in emotion_results:
                if layer not in all_results:
                    all_results[layer] = {}
                for loc in emotion_results[layer]:
                    if loc not in all_results[layer]:
                        all_results[layer][loc] = {}
                    for token in emotion_results[layer][loc]:
                        if token not in all_results[layer][loc]:
                            all_results[layer][loc][token] = {}
                        all_results[layer][loc][token][emotion] = emotion_results[layer][loc][token]
            
            # Clear memory
            del hidden_states
            if val_hidden_states is not None:
                del val_hidden_states
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Create summary DataFrame
        summary_df = self._summarize_probe_results(
            results=all_results,
            emotions_list=emotions_list,
            extraction_layers=extraction_layers,
            extraction_locs=extraction_locs,
            extraction_tokens=extraction_tokens
        )
        
        # Save summary
        summary_path = os.path.join(output_dir, 'probe_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        if self.logger:
            self.logger.info(f"Saved summary to: {summary_path}")
        
        # Save results by location if requested
        if save_by_location:
            if self.logger:
                self.logger.info("Saving results by location...")
            
            for loc in extraction_locs:
                loc_results = {}
                for layer in extraction_layers:
                    if layer not in all_results:
                        continue
                    if loc not in all_results[layer]:
                        continue
                    loc_results[layer] = {}
                    for token in extraction_tokens:
                        if token not in all_results[layer][loc]:
                            continue
                        loc_results[layer][token] = all_results[layer][loc][token]
                
                loc_results_path = os.path.join(
                    output_dir, 
                    f'binary_ova_probes_loc{loc}_{_probe_suffix_safe(extraction_layers, extraction_locs, extraction_tokens, model_name=self.model_short_name or self.hf_model_name)}.pt'
                )
                torch.save(loc_results, loc_results_path)
                if self.logger:
                    self.logger.info(f"Saved loc {loc} results")
        
        # Save combined results (filesystem-safe filename for Windows)
        final_results_path = os.path.join(
            output_dir, 
            f'binary_ova_probes_{_probe_suffix_safe(extraction_layers, extraction_locs, extraction_tokens, model_name=self.model_short_name or self.hf_model_name)}.pt'
        )
        torch.save(all_results, final_results_path)
        manifest_path = os.path.join(output_dir, "probe_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "model_id": self.model_short_name or self.hf_model_name,
                    "hf_model_name": self.hf_model_name,
                    "filename": os.path.basename(final_results_path),
                    "summary_filename": "probe_summary.csv",
                    "extraction_layers": list(extraction_layers),
                    "extraction_locs": list(extraction_locs),
                    "extraction_tokens": list(extraction_tokens),
                },
                f,
                indent=2,
            )
        if self.logger:
            self.logger.info(f"Saved combined results: {final_results_path}")
            self.logger.info(f"Saved probe manifest: {manifest_path}")
            self.logger.info("Training completed!")
        
        return {
            'results': all_results,
            'summary_df': summary_df,
            'output_dir': output_dir
        }
