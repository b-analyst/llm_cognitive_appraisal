import logging
import os
import subprocess
from datetime import datetime

from functools import partial
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from sklearn.linear_model import ElasticNet, LogisticRegression, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
class HookPoint:  # compatibility shim for old type annotations
    pass
import random


# Define the dataset class for handling text data
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        # self.labels = labels
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        return text, label


# Log class to handle logging activities
class Log:
    def __init__(self, log_name='probe'):
        filename = f'{log_name}_date-{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.txt'
        os.makedirs('logs', exist_ok=True)
        self.log_path = os.path.join('logs/', filename)
        self.logger = self._setup_logging()

    def _setup_logging(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(self.log_path),
                                logging.StreamHandler()
                            ])
        return logging.getLogger()


def log_system_info(logger):
    """
    Logs system memory and GPU details.
    """

    def run_command(command):
        """
        Runs a shell command and returns its output.

        Args:
        - command (list): Command and arguments to execute.

        Returns:
        - str: Output of the command.
        """
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return result.stderr

    gpu_info = run_command(['nvidia-smi'])

    if os.name == 'nt':  # windows system
        pass
    else:
        memory_info = run_command(['free', '-h'])
        logger.info("Memory Info:\n" + memory_info)

    logger.info("GPU Info:\n" + gpu_info)


def hf_login(logger):
    load_dotenv()
    try:
        # Retrieve the token from an environment variable
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token is None:
            logger.error("Hugging Face token not set in environment variables.")
            return

        # Attempt to log in with the Hugging Face token
        login(token=token)
        logger.info("Logged in successfully to Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")


def find_token_length_distribution(data, tokenizer):
    token_lengths = []
    for text in data:
        tokens = tokenizer.tokenize(text)
        token_lengths.append(len(tokens))

    token_lengths = np.array(token_lengths)
    quartiles = np.percentile(token_lengths, [25, 50, 75])
    min_length = np.min(token_lengths)
    max_length = np.max(token_lengths)

    return {
        "min_length": min_length,
        "25th_percentile": quartiles[0],
        "median": quartiles[1],
        "75th_percentile": quartiles[2],
        "max_length": max_length
    }

def emotion_to_token_ids(emotion_labels, tokenizer):
    some_random_text = "Hello, I am a random text."
    new_batch = [f"{some_random_text} {label}" for label in emotion_labels]

    inputs = tokenizer(
        new_batch,
        padding='longest',
        truncation=False,
        return_tensors="pt",
    )
    label_ids = inputs['input_ids'][:, -1]
    return label_ids

def get_emotion_logits(dataloader, tokenizer, model, ids_to_pick = None, apply_argmax = False):

    probs = []

    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = tokenizer(
            batch_texts,
            padding='longest',
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        logits = outputs.logits.cpu()

        logits = logits[:, -1, :]
        if not (ids_to_pick is None):
            logits = logits[:, ids_to_pick]

        if apply_argmax:
            logits = torch.argmax(logits, dim=-1)

        probs.append(logits)

    probs = torch.cat(probs, dim=0)
    return probs


def probe(all_hidden_states, labels, appraisals, logger):
    if isinstance(all_hidden_states, torch.Tensor):
        all_hidden_states = all_hidden_states.cpu().numpy()

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    # Normalize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Y_emotion = labels[:, 0]
    Y_appraisals = labels[:, 1:]

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # Probing for emotion (classification)
    try:
        # logger.info(f"Feature matrix shape: {X.shape}")
        # logger.info(f"Target vector shape: {Y_emotion.shape}")

        cv_accuracies = cross_val_score(LogisticRegression(max_iter=2000), X, Y_emotion, cv=kfold, scoring='accuracy')
        classifier = LogisticRegression(max_iter=2000)
        classifier.fit(X, Y_emotion)  # Train on the entire dataset for full model training after CV
        training_accuracy = classifier.score(X, Y_emotion)

        logger.info(f"5-Fold CV Accuracy for emotion category: {cv_accuracies.mean():.4f} ± {cv_accuracies.std():.4f}")
        logger.info(f"Training Accuracy for emotion category: {training_accuracy:.4f}")

        results['emotion'] = {
            'cv_accuracy': cv_accuracies.mean(),
            'cv_std': cv_accuracies.std(),
            'training_accuracy': training_accuracy
        }
    except Exception as e:
        logger.error(f"Error while probing emotion category: {e}")

    # Probing for each appraisal (regression)
    for i, appraisal_name in enumerate(appraisals):
        try:
            Y = Y_appraisals[:, i]
            logger.info(f"Probing appraisal: {appraisal_name}")
            # logger.info(f"Feature matrix shape: {X.shape}")
            # logger.info(f"Target vector shape: {Y.shape}")
            # logger.info(f"Feature 1st 5: {X[:5]}")
            # logger.info(f"Target 1st 5: {Y[:5]}")

            # Define parameter grid for ElasticNet
            param_grid = {
                'alpha': [0.1], #, 1.0, 10.0
                'l1_ratio': [0.1] #, 0.5, 0.9
            }
            
            enet = ElasticNet(max_iter=5000)
            grid_search = GridSearchCV(enet, param_grid, cv=kfold, scoring='r2', n_jobs=-1)
            grid_search.fit(X, Y)
            # enet.fit(X, Y)
            # best_model  = enet
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best hyperparameters for '{appraisal_name}': {best_params}")

            cv_mse = cross_val_score(best_model, X, Y, cv=kfold, scoring='neg_mean_squared_error')
            cv_r2 = cross_val_score(best_model, X, Y, cv=kfold, scoring='r2')
            
            training_predictions = best_model.predict(X)
            training_mse = mean_squared_error(Y, training_predictions)
            training_r2 = r2_score(Y, training_predictions)


            logger.info(f"5-Fold CV MSE for '{appraisal_name}': {-cv_mse.mean():.4f} ± {cv_mse.std():.4f}")
            logger.info(f"Training MSE for '{appraisal_name}': {training_mse:.4f}")
            logger.info(f"5-Fold CV R-squared for '{appraisal_name}': {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
            logger.info(f"Training R-squared for '{appraisal_name}': {training_r2:.4f}")
            logger.info("- -"*25)
        except Exception as e:
            logger.error(f"Error while probing appraisal '{appraisal_name}': {e}")
        
        results[appraisal_name] = {
            'training_mse': training_mse,
            'cv_mse': -cv_mse.mean(),
            'cv_mse_std': cv_mse.std(),
            'training_r2': training_r2,
            'cv_r2': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }

    return results

def probe_binary_relevance(all_hidden_states, labels, emotions_list, return_weights=False, Normalize_X=True,
                           C_grid=[0.01, 0.1, 1.0, 10.0], max_iter=5000, random_state=42, n_jobs=-1,
                           wandb_log=False, wandb_context=None, cv_folds=5):
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    try:
        import wandb
    except Exception:
        wandb = None
        wandb_log = False

    if isinstance(all_hidden_states, torch.Tensor):
        X = all_hidden_states.cpu().numpy().reshape(all_hidden_states.shape[0], -1)
    else:
        X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    if isinstance(labels, torch.Tensor):
        y = labels.cpu().numpy()
    else:
        y = np.asarray(labels)

    per_class = {}

    for cls_idx, cls_name in enumerate(emotions_list):
        y_bin = (y == cls_idx).astype(int)
        pos = int(y_bin.sum())
        neg = int(len(y_bin) - pos)
        print(f"Pos: {pos}, Neg: {neg}")
        if min(pos, neg) < 2:
            # not enough data to train/evaluate this class robustly
            continue
        print(y_bin)
        splits = min(cv_folds, pos, neg) 
        print(f"Using {splits} folds for {cls_name}")
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

        steps = []
        if Normalize_X:
            steps.append(('scaler', StandardScaler()))
        steps.append(('clf', LogisticRegression(
            solver='lbfgs',
            penalty='l2',
            class_weight='balanced',
            max_iter=max_iter,
            tol=1e-3,
            random_state=random_state,
        )))
        pipe = Pipeline(steps)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid={'clf__C': C_grid},
            scoring='roc_auc',
            cv=skf,
            n_jobs=n_jobs if n_jobs is not None else -1,  # INNER parallel
            refit=True,
            return_train_score=False
        )
        grid.fit(X, y_bin)

        cv_roc_auc = float(grid.best_score_)
        best_C = float(grid.best_params_['clf__C'])

        y_prob = grid.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        clf = grid.best_estimator_.named_steps['clf']
        entry = {
            'cv_roc_auc': cv_roc_auc,
            'train_accuracy': float(accuracy_score(y_bin, y_pred)),
            'train_f1': float(f1_score(y_bin, y_pred, zero_division=0)),
            'train_roc_auc': float(roc_auc_score(y_bin, y_prob)),
            'train_pr_auc': float(average_precision_score(y_bin, y_prob)),
            'best_C': best_C,
            "coef": clf.coef_.ravel(),
            "intercept": clf.intercept_.ravel()[0],
        }
        per_class[cls_name] = entry
        print(f"{cls_name}: {entry}")

        if wandb_log and (wandb is not None) and (wandb.run is not None):
            log_row = {
                'cv_roc_auc': entry['cv_roc_auc'],   # single metric key for all classes
                'emotion_name': cls_name,            # use as stroke in a Line Plot
                'best_C': entry['best_C'],
                'train_f1': entry['train_f1'],
            }
            if isinstance(wandb_context, dict):
                log_row.update(wandb_context)
            wandb.log(log_row)  # no explicit step

    W = np.stack([entry['coef'] for entry in per_class.values()], axis=0)
    b = np.asarray([entry['intercept'] for entry in per_class.values()])

    out = {'per_class': per_class}
    if return_weights:
        out['weights'] = W
        out['bias'] = b
    return out

def probe_regression(all_hidden_states, labels, return_weights=False):
    if len(labels.shape) == 1:
        labels = labels[:, None]

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    Y = labels

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # scaler = StandardScaler(with_std=False)
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # scaler = StandardScaler()
    # Y_train = scaler.fit_transform(Y_train)
    # Y_test = scaler.transform(Y_test)

    net = Ridge(alpha=5.0)  # ElasticNet(alpha=0.1, l1_ratio=0.1)
    net.fit(X_train, Y_train)
    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)

    mse_train = mean_squared_error(Y_train, y_pred_train)
    mse_test = mean_squared_error(Y_test, y_pred_test)
    r2_train = r2_score(Y_train, y_pred_train)
    r2_test = r2_score(Y_test, y_pred_test)
    res = {'mse_train': mse_train, 'mse_test': mse_test, 'r2_train': r2_train, 'r2_test': r2_test}
    if return_weights:
        res['weights'] = net.coef_
        res['bias'] = net.intercept_
    return res


def probe_classification(all_hidden_states, labels, return_weights=False, Normalize_X = False, reg_strength = 1.0, fit_intercept = True):
    if len(labels.shape) == 2:
        labels = labels[:, 0]

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    Y = labels

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    if Normalize_X:
        scaler = StandardScaler(with_std=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    net = LogisticRegression(C = 1 / reg_strength, fit_intercept=fit_intercept)
    net.fit(X_train, Y_train)

    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)

    if isinstance(Y_train, np.ndarray):
        Y_train = torch.tensor(Y_train)
        Y_test = torch.tensor(Y_test)
    
    if isinstance(y_pred_train, np.ndarray):
        y_pred_train = torch.tensor(y_pred_train)
        y_pred_test = torch.tensor(y_pred_test)
    
    accuracy_train = (Y_train == y_pred_train).float().mean()

    accuracy_test = (Y_test == y_pred_test).float().mean()
    res = {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test}
    if return_weights:
        res['weights'] = net.coef_
        res['bias'] = net.intercept_

    return res

def probe_classification_non_linear(all_hidden_states, labels, return_weights=False, Normalize_X = False, reg_strength = 1.0, fit_intercept = True):
    if len(labels.shape) == 2:
        labels = labels[:, 0]

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    Y = labels

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    if Normalize_X:
        scaler = StandardScaler(with_std=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    net = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, activation='relu')
    net.fit(X_train, Y_train)

    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)

    if isinstance(Y_train, np.ndarray):
        Y_train = torch.tensor(Y_train)
        Y_test = torch.tensor(Y_test)
    
    if isinstance(y_pred_train, np.ndarray):
        y_pred_train = torch.tensor(y_pred_train)
        y_pred_test = torch.tensor(y_pred_test)
    
    accuracy_train = (Y_train == y_pred_train).float().mean()

    accuracy_test = (Y_test == y_pred_test).float().mean()
    res = {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test}
    
    return res



extraction_locations = {1: "model.layers.[LID].hook_initial_hs",
                        2: "model.layers.[LID].hook_after_attn_normalization",
                        3: "model.layers.[LID].hook_after_attn",
                        4: "model.layers.[LID].hook_after_attn_hs",
                        5: "model.layers.[LID].hook_after_mlp_normalization",
                        6: "model.layers.[LID].hook_after_mlp",
                        7: "model.layers.[LID].hook_after_mlp_hs",
                        8: "model.layers.[LID].self_attn.hook_attn_heads",
                        9: "model.final_hook",
                        10: "model.layers.[LID].self_attn.hook_attn_weights",
                        }


class _NamedHookRef:
    def __init__(self, name: str):
        self.name = name


def _uses_transformer_lens_hooks(model) -> bool:
    return hasattr(model, "run_with_cache") and hasattr(model, "run_with_hooks")


def _layer_module(model, layer_idx: int):
    return model.model.layers[layer_idx]


def _module_path_for_loc(model, layer_idx: int, loc: int):
    layer = _layer_module(model, layer_idx)
    if loc == 1:
        return layer, "pre"
    if loc == 2 and hasattr(layer, "input_layernorm"):
        return layer.input_layernorm, "forward"
    if loc == 3 and hasattr(layer, "self_attn"):
        return layer.self_attn, "forward"
    if loc == 5 and hasattr(layer, "post_attention_layernorm"):
        return layer.post_attention_layernorm, "forward"
    if loc == 6 and hasattr(layer, "mlp"):
        return layer.mlp, "forward"
    if loc == 7:
        return layer, "forward"
    if loc == 8 and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn.o_proj, "pre"
    return None, None


def _extract_tensor_from_hook_value(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        first = value[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


def _replace_tensor_in_hook_value(value, new_tensor):
    if isinstance(value, torch.Tensor):
        return new_tensor
    if isinstance(value, tuple) and value:
        return (new_tensor,) + tuple(value[1:])
    if isinstance(value, list) and value:
        return [new_tensor] + list(value[1:])
    return value


def _build_stock_hook_specs(model, extraction_layers, extraction_locs):
    specs = []
    for layer in extraction_layers:
        for loc in extraction_locs:
            name = extraction_locations[loc].replace("[LID]", str(layer))
            module, hook_kind = _module_path_for_loc(model, layer, loc)
            if module is None:
                raise NotImplementedError(
                    f"Stock hook backend does not support loc={loc} for model {type(model).__name__}."
                )
            specs.append(
                {
                    "name": name,
                    "layer": layer,
                    "loc": loc,
                    "module": module,
                    "hook_kind": hook_kind,
                }
            )
    return specs


def _run_stock_model_with_cache(model, inputs, capture_specs, output_attentions=False, intervention_spec=None):
    cache_dict = {}
    handles = []
    intervention_spec = intervention_spec or {}

    def _apply_intervention(tensor, spec):
        if isinstance(spec, tuple) and len(spec) == 2:
            vec, strength = spec
            op = "add"
        else:
            vec = spec.get("vector")
            strength = spec.get("strength", 1.0)
            op = spec.get("op", "add")
        if isinstance(vec, torch.Tensor):
            vec_t = vec.to(device=tensor.device, dtype=tensor.dtype)
        else:
            vec_t = torch.tensor(vec, dtype=tensor.dtype, device=tensor.device)
        vec_t = vec_t.reshape(-1)
        out = tensor.clone()
        if op == "add":
            out[:, -1, :] = out[:, -1, :] + strength * vec_t
        elif op == "erase":
            x = out[:, -1, :].float()
            d = vec_t.float()
            d_sq = (d @ d).clamp(min=1e-8)
            coef = (x @ d) / d_sq
            x = x - strength * coef.unsqueeze(1) * d.unsqueeze(0)
            out[:, -1, :] = x.to(out.dtype)
        else:
            raise ValueError(f"Unsupported intervention op: {op}")
        return out

    def _make_forward_hook(spec):
        name = spec["name"]
        layer = spec["layer"]
        loc = spec["loc"]

        def hook(_module, _inp, output):
            tensor = _extract_tensor_from_hook_value(output)
            if tensor is None:
                return output
            maybe_new = tensor
            if (layer, loc) in intervention_spec:
                maybe_new = _apply_intervention(tensor, intervention_spec[(layer, loc)])
            cached = maybe_new.detach()
            if cached.is_floating_point():
                cached = cached.float()
            cache_dict[name] = cached.cpu()
            if (layer, loc) in intervention_spec:
                return _replace_tensor_in_hook_value(output, maybe_new)
            return output

        return hook

    def _make_pre_hook(spec):
        name = spec["name"]
        layer = spec["layer"]
        loc = spec["loc"]

        def hook(_module, args):
            if not args:
                return args
            tensor = _extract_tensor_from_hook_value(args)
            if tensor is None:
                return args
            maybe_new = tensor
            if (layer, loc) in intervention_spec:
                maybe_new = _apply_intervention(tensor, intervention_spec[(layer, loc)])
            cached = maybe_new.detach()
            if cached.is_floating_point():
                cached = cached.float()
            cache_dict[name] = cached.cpu()
            if (layer, loc) in intervention_spec:
                return (maybe_new,) + tuple(args[1:])
            return args

        return hook

    for spec in capture_specs:
        if spec["hook_kind"] == "pre":
            handles.append(spec["module"].register_forward_pre_hook(_make_pre_hook(spec), with_kwargs=False))
        else:
            handles.append(spec["module"].register_forward_hook(_make_forward_hook(spec)))

    try:
        outputs = model(**inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)
    finally:
        for h in handles:
            h.remove()

    if output_attentions and getattr(outputs, "attentions", None) is not None:
        for spec in capture_specs:
            if spec["loc"] == 10:
                attn = outputs.attentions[spec["layer"]]
                if attn is not None:
                    cached = attn.detach()
                    if cached.is_floating_point():
                        cached = cached.float()
                    cache_dict[spec["name"]] = cached.cpu()
    return outputs, cache_dict


def _run_stock_generate_with_intervention(
    model,
    inputs,
    intervention_spec=None,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    intervention_during_decode=True,
):
    """Run model.generate with the same stock-hook intervention mechanism used for extraction.

    Parameters
    ----------
    intervention_during_decode
        If True (default), apply the intervention on every forward pass during generate()
        (prefill and each new token). If False, skip when the activation sequence length is 1,
        which typically corresponds to autoregressive decode steps with KV cache — so steering
        applies during prompt prefill only. This reduces repetitive corruption of hidden states
        during long generations.
    """
    intervention_spec = intervention_spec or {}
    handles = []

    def _normalize_token_id(value):
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    def _apply_intervention(tensor, spec):
        if isinstance(spec, tuple) and len(spec) == 2:
            vec, strength = spec
            op = "add"
        else:
            vec = spec.get("vector")
            strength = spec.get("strength", 1.0)
            op = spec.get("op", "add")
        if isinstance(vec, torch.Tensor):
            vec_t = vec.to(device=tensor.device, dtype=tensor.dtype)
        else:
            vec_t = torch.tensor(vec, dtype=tensor.dtype, device=tensor.device)
        vec_t = vec_t.reshape(-1)
        out = tensor.clone()
        if op == "add":
            out[:, -1, :] = out[:, -1, :] + strength * vec_t
        elif op == "erase":
            x = out[:, -1, :].float()
            d = vec_t.float()
            d_sq = (d @ d).clamp(min=1e-8)
            coef = (x @ d) / d_sq
            x = x - strength * coef.unsqueeze(1) * d.unsqueeze(0)
            out[:, -1, :] = x.to(out.dtype)
        else:
            raise ValueError(f"Unsupported intervention op: {op}")
        return out

    if intervention_spec:
        layers = sorted({int(layer) for layer, _ in intervention_spec.keys()})
        locs = sorted({int(loc) for _, loc in intervention_spec.keys()})
        hook_specs = _build_stock_hook_specs(model, layers, locs)

        def _should_apply(tensor) -> bool:
            if tensor is None or not hasattr(tensor, "shape") or len(tensor.shape) < 2:
                return True
            if intervention_during_decode:
                return True
            # Prefill uses full prompt length; decode-with-cache usually passes seq_len==1.
            return int(tensor.shape[1]) != 1

        def _make_forward_hook(spec):
            layer = spec["layer"]
            loc = spec["loc"]

            def hook(_module, _inp, output):
                tensor = _extract_tensor_from_hook_value(output)
                if tensor is None or (layer, loc) not in intervention_spec:
                    return output
                if not _should_apply(tensor):
                    return output
                maybe_new = _apply_intervention(tensor, intervention_spec[(layer, loc)])
                return _replace_tensor_in_hook_value(output, maybe_new)

            return hook

        def _make_pre_hook(spec):
            layer = spec["layer"]
            loc = spec["loc"]

            def hook(_module, args):
                if not args or (layer, loc) not in intervention_spec:
                    return args
                tensor = _extract_tensor_from_hook_value(args)
                if tensor is None:
                    return args
                if not _should_apply(tensor):
                    return args
                maybe_new = _apply_intervention(tensor, intervention_spec[(layer, loc)])
                return (maybe_new,) + tuple(args[1:])

            return hook

        for spec in hook_specs:
            if spec["hook_kind"] == "pre":
                handles.append(spec["module"].register_forward_pre_hook(_make_pre_hook(spec), with_kwargs=False))
            else:
                handles.append(spec["module"].register_forward_hook(_make_forward_hook(spec)))

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "pad_token_id": _normalize_token_id(getattr(model.config, "pad_token_id", None)),
    }
    if gen_kwargs["pad_token_id"] is None:
        gen_kwargs["pad_token_id"] = _normalize_token_id(getattr(model.config, "eos_token_id", None))
    if do_sample:
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["temperature"] = 1.0

    try:
        with torch.no_grad():
            sequences = model.generate(**inputs, **gen_kwargs)
    finally:
        for h in handles:
            h.remove()
    return sequences

def name_to_loc_and_layer(name):
    layer = int(name.split("model.layers.")[1].split(".")[0])
    loc_suffixes = {v.split('.')[-1]:k for k,v in extraction_locations.items()}
    loc = loc_suffixes[name.split(".")[-1]]
    
    return loc, layer

def extract_from_cache(cache_dict_, extraction_layers=[0, 1],
                          extraction_locs=[1, 7],
                          extraction_tokens=[-1]):
    return_value = []

    for layer in extraction_layers:
        return_value.append([])
        for el_ in extraction_locs:
            el = extraction_locations[el_].replace("[LID]", str(layer))
            if el_ != 10: # attention weights should be treated differently
                t = cache_dict_[el][:, extraction_tokens]
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    t = t.float()
                return_value[-1].append(t.cpu() if isinstance(t, torch.Tensor) else t)
            else:
                t = cache_dict_[el][:, :, extraction_tokens]
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    t = t.float()
                return_value[-1].append(t.cpu() if isinstance(t, torch.Tensor) else t)

        return_value[-1] = torch.stack(return_value[-1], dim=1)
    return_value = torch.stack(return_value, dim=1)
    return return_value


def _gather_mid_token_hidden(t_full, attention_mask):
    """
    Per-row masked middle index on the sequence axis of t_full.

    t_full: (batch, seq_len, hidden)
    attention_mask: (batch, seq_len), 1 for real (non-pad) positions.

    first_real = smallest index with mask>0; last_real = sum(mask)-1 (clamped).
    mid = (first_real + last_real) // 2. Empty rows fall back to index 0.
    """
    device = t_full.device
    mask = attention_mask.to(device=device, dtype=torch.long)
    batch, seq_len, _ = t_full.shape
    idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    big = seq_len + 64
    first_real = torch.where(mask > 0, idx, torch.full_like(idx, big)).min(dim=1).values
    sum_m = mask.sum(dim=1)
    empty = sum_m == 0
    last_real = (sum_m - 1).clamp(min=0)
    first_real = torch.where(empty, torch.zeros_like(first_real), first_real.clamp(0, seq_len - 1))
    mid = ((first_real + last_real) // 2).clamp(0, seq_len - 1)
    mid = torch.where(empty, torch.zeros_like(mid), mid)
    b_idx = torch.arange(batch, device=device, dtype=torch.long)
    return t_full[b_idx, mid, :]


def extract_from_cache_mixed(
    cache_dict_,
    extraction_layers,
    extraction_locs,
    extraction_tokens,
    attention_mask,
):
    """
    Same layout as extract_from_cache (batch, n_layers, n_locs, n_token_slots, dim) but
    allows extraction_tokens entries to include the string 'mid' for per-sequence middle gather.
    Does not support loc==10 (attention weights).
    """
    return_value = []
    for layer in extraction_layers:
        return_value.append([])
        for el_ in extraction_locs:
            el = extraction_locations[el_].replace("[LID]", str(layer))
            if el_ == 10:
                raise ValueError("extract_from_cache_mixed does not support attention extraction (loc 10).")
            t_full = cache_dict_[el]
            slots = []
            for tok in extraction_tokens:
                if isinstance(tok, str) and tok.lower() == "mid":
                    vec = _gather_mid_token_hidden(t_full, attention_mask)
                else:
                    ti = int(tok)
                    vec = t_full[:, ti, :]
                slots.append(vec.unsqueeze(1))
            stacked = torch.cat(slots, dim=1)
            if isinstance(stacked, torch.Tensor) and stacked.is_floating_point():
                stacked = stacked.float()
            return_value[-1].append(stacked.cpu() if isinstance(stacked, torch.Tensor) else stacked)

        return_value[-1] = torch.stack(return_value[-1], dim=1)
    return_value = torch.stack(return_value, dim=1)
    return return_value


def extract_hidden_states(dataloader, tokenizer, model, logger,
                          extraction_layers=[0, 1],
                          extraction_locs=[1, 7],
                          extraction_tokens=[-1],
                          do_final_cat = True, return_tokenized_input = False):
    assert [extraction_loc in extraction_locations.keys() for extraction_loc in extraction_locs]    
    assert (10 not in extraction_locs) or len(extraction_locs) == 1
        
    output_attentions = 10 in extraction_locs

    return_values = []
    
    tokenized_input = []

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Pre-determine max sequence length if using auto mode
    if extraction_tokens == 'auto':
        max_length = 0
        for batch_texts, _ in dataloader:
            # Add padding to handle variable lengths within batch during pre-scan
            temp_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=258)
            max_length = max(max_length, temp_inputs['input_ids'].shape[1])
        # Add some buffer but cap at memory-safe limit
        max_length = min(max_length + 10, 128)  # Cap at 128 tokens to prevent VRAM explosion
    else:
        max_length = None
    
    stock_specs = None if _uses_transformer_lens_hooks(model) else _build_stock_hook_specs(model, extraction_layers, extraction_locs)

    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):

        inputs = tokenizer(
            batch_texts,
            padding='max_length' if max_length else 'longest',
            max_length=max_length,
            truncation=True,  # Add truncation to prevent super long sequences
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            if _uses_transformer_lens_hooks(model):
                outputs = model.run_with_cache(**inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)
                cache_dict_ = outputs[1]
            else:
                outputs, cache_dict_ = _run_stock_model_with_cache(
                    model, inputs, stock_specs, output_attentions=output_attentions
                )

        # Determine extraction tokens - now all sequences are padded to same length
        if extraction_tokens == 'auto' or extraction_tokens is None:
            # Extract all token positions since they're all padded to same length
            seq_length = inputs['input_ids'].shape[1]
            current_extraction_tokens = list(range(seq_length))
            r = extract_from_cache(cache_dict_, extraction_layers=extraction_layers,
                                   extraction_locs=extraction_locs,
                                   extraction_tokens=current_extraction_tokens)
        else:
            current_extraction_tokens = extraction_tokens
            if (
                isinstance(current_extraction_tokens, list)
                and any(isinstance(t, str) and str(t).lower() == "mid" for t in current_extraction_tokens)
            ):
                attn = inputs["attention_mask"]
                r = extract_from_cache_mixed(
                    cache_dict_,
                    extraction_layers=extraction_layers,
                    extraction_locs=extraction_locs,
                    extraction_tokens=current_extraction_tokens,
                    attention_mask=attn,
                )
            else:
                r = extract_from_cache(
                    cache_dict_,
                    extraction_layers=extraction_layers,
                    extraction_locs=extraction_locs,
                    extraction_tokens=current_extraction_tokens,
                )
        
        return_values.append(r)
        
        if return_tokenized_input:
            assert len(inputs['input_ids']) == 1, "Batch size must be 1 for tokenized input extraction"
            tokenized_input.append(tokenizer.convert_ids_to_tokens([w for w in inputs['input_ids'][0].cpu()]))

    if do_final_cat:
        return_values = torch.cat(return_values, dim=0)
    
    if return_tokenized_input:
        return return_values, tokenized_input
    return return_values


def run_forward_with_steering(
    dataloader,
    tokenizer,
    model,
    extraction_layers,
    extraction_locs,
    extraction_tokens,
    steering_spec,
    logger=None,
    show_progress=True,
):
    """
    Run model forward with optional steering at (layer, loc) hook points; capture
    hidden states at extraction layers/locs for downstream probe readout.

    steering_spec: dict mapping (layer, loc) -> (vector_tensor, strength).
        Vector is (hidden_size,) or (1, hidden_size). At each such (layer, loc),
        we add strength * vector to the last token position.
    extraction_*: same as extract_hidden_states (which layers/locs/tokens to capture).

    Returns:
        hidden_states: (n_samples, n_layers, n_locs, n_tokens, dim) like extract_hidden_states.
        outputs_list: list of model output objects (one per batch) for e.g. logits.
    """
    assert [loc in extraction_locations.keys() for loc in extraction_locs]
    assert (10 not in extraction_locs), "Cannot capture attention weights"
    if isinstance(extraction_tokens, int):
        extraction_tokens = [extraction_tokens]

    names_to_capture = [
        extraction_locations[loc].replace("[LID]", str(layer))
        for layer in extraction_layers
        for loc in extraction_locs
    ]
    names_to_capture_set = set(names_to_capture)

    # steering_spec: (layer, loc) -> (vec, strength); vec on CPU or same device as model
    cache_dict = {}
    stock_specs = None if _uses_transformer_lens_hooks(model) else _build_stock_hook_specs(model, extraction_layers, extraction_locs)

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def steering_and_cache_hook(act, hook):
        name = hook.name
        if name not in names_to_capture_set:
            return act
        loc, layer = name_to_loc_and_layer(name)
        if (layer, loc) in steering_spec:
            spec = steering_spec[(layer, loc)]
            if isinstance(spec, tuple) and len(spec) == 2:
                vec, strength = spec
                op = "add"
            else:
                vec = spec.get("vector")
                strength = spec.get("strength", 1.0)
                op = spec.get("op", "add")
            if isinstance(vec, np.ndarray):
                vec = torch.tensor(vec, dtype=act.dtype, device=act.device)
            elif isinstance(vec, torch.Tensor):
                vec = vec.to(act.device, dtype=act.dtype)
            else:
                vec = torch.tensor(vec, dtype=act.dtype, device=act.device)
            vec = vec.reshape(-1)
            act = act.clone()
            if op == "add":
                act[:, -1, :] = act[:, -1, :] + strength * vec
            elif op == "erase":
                x = act[:, -1, :].float()
                d = vec.float()
                d_sq = (d @ d).clamp(min=1e-8)
                coef = (x @ d) / d_sq
                x = x - strength * coef.unsqueeze(1) * d.unsqueeze(0)
                act[:, -1, :] = x.to(act.dtype)
            else:
                raise ValueError(f"Unsupported intervention op: {op}")
        cache_dict[name] = act.cpu()
        return act

    return_values = []
    outputs_list = []
    n_batches = len(dataloader)
    desc = "Forward pass (steering)"
    for i, (batch_texts, _) in enumerate(tqdm(dataloader, total=n_batches, desc=desc, leave=True, disable=not show_progress)):
        if logger and i == 0 and show_progress:
            logger.info("Behavioral steering: running model with in-pass steering hooks...")
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)
        cache_dict.clear()
        with torch.no_grad():
            if _uses_transformer_lens_hooks(model):
                def run_for_capture(name_or_hook):
                    name = name_or_hook if isinstance(name_or_hook, str) else getattr(name_or_hook, "name", None)
                    return name in names_to_capture_set
                outputs = model.run_with_hooks(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=True,
                    fwd_hooks=[(run_for_capture, steering_and_cache_hook)],
                )
            else:
                outputs, cache_dict = _run_stock_model_with_cache(
                    model,
                    inputs,
                    stock_specs,
                    output_attentions=False,
                    intervention_spec=steering_spec,
                )
        outputs_list.append(outputs)
        r = extract_from_cache(
            cache_dict,
            extraction_layers=extraction_layers,
            extraction_locs=extraction_locs,
            extraction_tokens=extraction_tokens,
        )
        return_values.append(r)
    return torch.cat(return_values, dim=0), outputs_list


def generate_with_steering(
    dataloader,
    tokenizer,
    model,
    steering_spec,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    logger=None,
    show_progress=True,
    intervention_during_decode=True,
):
    """
    Run actual text generation with optional steering at (layer, loc) hook points.

    intervention_during_decode
        Passed to `_run_stock_generate_with_intervention`. Set False to steer only during
        prompt prefill (typical seq_len > 1), not on each decode step (seq_len == 1).

    Returns a list of dict rows with prompt text, generated continuation, and full text.
    """
    if _uses_transformer_lens_hooks(model):
        raise NotImplementedError("generate_with_steering currently supports the stock Hugging Face hook backend only.")

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    n_batches = len(dataloader)
    desc = "Generation (steering)"
    for i, (batch_texts, _) in enumerate(tqdm(dataloader, total=n_batches, desc=desc, leave=True, disable=not show_progress)):
        if logger and i == 0 and show_progress:
            logger.info("Behavioral generation: running model.generate with steering hooks...")
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)
        input_len = int(inputs["input_ids"].shape[1])
        sequences = _run_stock_generate_with_intervention(
            model,
            inputs,
            intervention_spec=steering_spec,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            intervention_during_decode=intervention_during_decode,
        )
        sequences = sequences.detach().cpu()
        for prompt_text, seq in zip(batch_texts, sequences):
            continuation_ids = seq[input_len:]
            generated_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            full_text = tokenizer.decode(seq, skip_special_tokens=True).strip()
            rows.append(
                {
                    "prompt_text": str(prompt_text),
                    "generated_text": generated_text,
                    "full_text": full_text,
                    "n_generated_tokens": int(continuation_ids.shape[0]),
                }
            )
    return rows


def apply_zero_intervention_and_extract_logits(dataloader, tokenizer, model, logger,
                                               intervention_layers=[0, 1], intervention_tokens='all',
                                               intervention_locs=[1, 7],
                                               ids_to_pick=None):
    assert [intervention_loc in extraction_locations.keys() for intervention_loc in intervention_locs]
    intervention_tokens = intervention_tokens if intervention_tokens != 'all' else slice(None)
    names_to_intervene = [extraction_locations[loc].replace("[LID]", str(layer)) for layer in intervention_layers
                            for loc in intervention_locs]
    
    def zero_intervention_hook(input_vector, hook: HookPoint):
        name = hook.name

        if name in names_to_intervene:
            input_vector[:, intervention_tokens] = input_vector[:, intervention_tokens] * 0

        return input_vector

    returned_logits = []
    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = tokenizer(
            batch_texts,
            padding='longest',
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.run_with_hooks(**inputs, return_dict=True, output_hidden_states=True,
                                           fwd_hooks=[(lambda x: True, zero_intervention_hook)])

        logits = outputs.logits.cpu()
        logits = logits[:, -1, :]
        if not (ids_to_pick is None):
            logits = logits[:, ids_to_pick]

        returned_logits.append(logits)

    return torch.cat(returned_logits, dim=0)

def apply_random_intervention_and_extract_logits(dataloader, tokenizer, model, logger,
                                               intervention_layers=[0, 1], intervention_tokens='all',
                                               intervention_locs=[1, 7],
                                               ids_to_pick=None):
    assert [intervention_loc in extraction_locations.keys() for intervention_loc in intervention_locs]
    intervention_tokens = intervention_tokens if intervention_tokens != 'all' else slice(None)
    names_to_intervene = [extraction_locations[loc].replace("[LID]", str(layer)) for layer in intervention_layers
                            for loc in intervention_locs]
    
    def random_intervention_hook(input_vector, hook: HookPoint):
        name = hook.name

        if name in names_to_intervene:
            v = input_vector[:, intervention_tokens]
            v_ = torch.randn_like(v)
            v_ = v_ / v_.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)
            
            input_vector[:, intervention_tokens] = v * 0 + v_

        return input_vector

    returned_logits = []
    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = tokenizer(
            batch_texts,
            padding='longest',
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.run_with_hooks(**inputs, return_dict=True, output_hidden_states=True,
                                           fwd_hooks=[(lambda x: True, random_intervention_hook)])

        logits = outputs.logits.cpu()
        logits = logits[:, -1, :]
        if not (ids_to_pick is None):
            logits = logits[:, ids_to_pick]

        returned_logits.append(logits)

    return torch.cat(returned_logits, dim=0)



def activation_patching(source_sentence, target_sentence, tokenizer, model, logger, intervention_layers=[0, 1],
                        intervention_locs=[1, 7], intervention_tokens=[-1], ids_to_pick=None):
    assert [intervention_loc in extraction_locations.keys() for intervention_loc in intervention_locs]

    source_sentence_ids = tokenizer([source_sentence], return_tensors="pt", padding='longest', truncation=False).to(
        model.device)
    target_sentence_ids = tokenizer([target_sentence], return_tensors="pt", padding='longest', truncation=False).to(
        model.device)

    with torch.no_grad():
        source_outputs = model.run_with_cache(**source_sentence_ids, return_dict=True, output_hidden_states=True)
        source_clean_cache = {k: v.cpu() for k, v in source_outputs[1].items()}
        source_clean_logits = source_outputs[0].logits[0, -1].cpu()
        del source_outputs

        target_outputs = model.run_with_cache(**target_sentence_ids, return_dict=True, output_hidden_states=True)
        target_clean_cache = {k: v.cpu() for k, v in target_outputs[1].items()}
        target_clean_logits = target_outputs[0].logits[0, -1].cpu()
        del target_outputs

    if not (ids_to_pick is None):
        source_clean_logits = source_clean_logits[ids_to_pick]
        target_clean_logits = target_clean_logits[ids_to_pick]

    def patching_hook(input_vector, hook: HookPoint):
        name = hook.name
        names_to_intervene = [extraction_locations[loc].replace("[LID]", str(layer)) for layer in intervention_layers
                              for loc in intervention_locs]
        if name in names_to_intervene:
            input_vector[:, intervention_tokens] = input_vector[:, intervention_tokens] * 0 + source_clean_cache[name][
                                                                                              :,
                                                                                              intervention_tokens].to(
                input_vector.device)
        return input_vector

    with torch.no_grad():
        outputs = model.run_with_hooks(**target_sentence_ids, return_dict=True, output_hidden_states=True,
                                       fwd_hooks=[(lambda x: True, patching_hook)])
        patched_logits = outputs.logits[0, -1].cpu()

    if not (ids_to_pick is None):
        patched_logits = patched_logits[ids_to_pick]

    return {'source_clean_logits': source_clean_logits, 'target_clean_logits': target_clean_logits,
            'patched_logits': patched_logits}



def promote_vec(dataloader, tokenizer, model, logger, prom_vector, projection_matrix, Beta,
                          promotion_layers  = [1, 2] , promotion_locs  =  [3, 6], promotion_tokens   = [-1],
                          extraction_layers = [0],  extraction_locs =  [7],  extraction_tokens = [-1],
                          ids_to_pick=None,):
        
        
        assert [extraction_loc in extraction_locations.keys() for extraction_loc in extraction_locs]    
        assert (10 not in extraction_locs), "Cannot extract attention weights from this function"
        
        hs = []
        
        hidden_state_size = model.config.hidden_size
        
        assert promotion_tokens == 'all' or isinstance(promotion_tokens, list)
        if promotion_tokens == 'all':
            promotion_tokens = slice(None)
            assert prom_vector.shape       == (len(promotion_layers), len(promotion_locs), 1,                     hidden_state_size)
            assert projection_matrix.shape == (len(promotion_layers), len(promotion_locs), 1,                     hidden_state_size, hidden_state_size)            
        else:
            assert prom_vector.shape       == (len(promotion_layers), len(promotion_locs), len(promotion_tokens), hidden_state_size)
            assert projection_matrix.shape == (len(promotion_layers), len(promotion_locs), len(promotion_tokens), hidden_state_size, hidden_state_size)            
        
        first_terms = prom_vector.unsqueeze(0) # add batch dimension
        
        names_to_promote = [extraction_locations[loc].replace("[LID]", str(layer)) for layer in promotion_layers for loc in promotion_locs]
    
        def promotion_hook(input_vector, hook: HookPoint, cache_dict: dict):
            
            
            
            name = hook.name
            
            
            if name in names_to_promote:
                loc, layer = name_to_loc_and_layer(name)
                
                layer_idx = promotion_layers.index(layer)
                loc_idx = promotion_locs.index(loc)
                
                uA = input_vector[:, promotion_tokens, :] # batch, token, hidden
                
                P = projection_matrix[layer_idx, loc_idx, :, :, :].to(uA.device)  # token, hidden, hidden
                
                first_term = first_terms[:, layer_idx, loc_idx].to(uA.device)
                second_term = Beta * torch.einsum('tdh, bth->btd', P, uA)
                
                uA = uA + first_term - second_term
                
                input_vector[:, promotion_tokens, :] = uA

            cache_dict[name] = input_vector.clone()
            
            return input_vector

        returned_logits = []
        
        for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = tokenizer(
                batch_texts,
                padding='longest',
                truncation=False,
                return_tensors="pt",
            ).to(model.device)

            
            with torch.no_grad():
                cache_dict_ = {}
                ph = partial(promotion_hook, cache_dict=cache_dict_)
                outputs = model.run_with_hooks(**inputs, return_dict=True, output_hidden_states=True,
                                            fwd_hooks=[(lambda x: True, ph)])

            r = extract_from_cache(cache_dict_, extraction_layers=extraction_layers, extraction_locs=extraction_locs, extraction_tokens=extraction_tokens)
            hs.append(r)
            
            logits = outputs.logits.cpu()
            logits = logits[:, -1, :]
            if not (ids_to_pick is None):
                logits = logits[:, ids_to_pick]

            returned_logits.append(logits)

        hs = torch.cat(hs, dim=0)
        return torch.cat(returned_logits, dim=0), hs


def make_projections(w):
    #assuming the last 2 dimensions of w, shows the number of vectors and the size of the vector dimension respectively
    w_shape = w.shape[:-2]
    n = w.shape[-2]
    d = w.shape[-1]
    
    #flatten except the last 2 dimensions
    w = w.reshape(-1, n, d)
    
    return_result = torch.zeros([w.shape[0], d, d], device=w.device, dtype=w.dtype)
    
    for i in range(w.shape[0]):
        w_ = w[i]
        return_result[i] = w_.T @ (w_ @ w_.T).inverse() @ w_
    
    return return_result.reshape(w_shape + (d, d))


def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def apply_classification_probe(data, weights, bias):
    num_data = data.shape[0]
    features_dim = data.shape[-1]
    num_classes = weights.shape[0]
    
    data_shape = data.shape[1:-1]
    data = data.reshape(num_data, -1, features_dim)
    
    weights = weights.reshape(num_classes, -1, features_dim)
    bias = bias.reshape(num_classes, -1)
    
    logits = torch.einsum('ctd, btd->btc', weights, data) + bias.T
    logits = logits.reshape(num_data, *data_shape, num_classes)
    
    return logits
    
    

def apply_regression_probe(data, weights, bias):
    num_data = data.shape[0]
    features_dim = data.shape[-1]
    num_outputs = weights.shape[0]
    
    data_shape = data.shape[1:-1]
    data = data.reshape(num_data, -1, features_dim)
    
    weights = weights.reshape(num_outputs, -1, features_dim)
    bias = bias.reshape(num_outputs, -1)
    
    logits = torch.einsum('ctd, btd->btc', weights, data) + bias.T
    logits = logits.reshape(num_data, *data_shape, num_outputs)
    
    return logits

def apply_probes_on_hs(hs, emotions_weights, emotions_biases, appraisals_weights, appraisals_biases,
                       extraction_layers=list(range(16)), extraction_locs=[3, 6, 7], extraction_tokens=[-1],
                       layer_to_monitor = 15, loc_to_monitor = 7, token_to_monitor = -1):
    w = emotions_weights[:, extraction_layers.index(layer_to_monitor), extraction_locs.index(loc_to_monitor), extraction_tokens.index(token_to_monitor), :]
    b = emotions_biases[:, extraction_layers.index(layer_to_monitor), extraction_locs.index(loc_to_monitor), extraction_tokens.index(token_to_monitor)]
    preds_emo = apply_classification_probe(hs, w, b)
    
    w = appraisals_weights[:, extraction_layers.index(layer_to_monitor), extraction_locs.index(loc_to_monitor), extraction_tokens.index(token_to_monitor), :]
    b = appraisals_biases[:, extraction_layers.index(layer_to_monitor), extraction_locs.index(loc_to_monitor), extraction_tokens.index(token_to_monitor)]
    preds_app = apply_regression_probe(hs, w, b)
    
    return preds_emo, preds_app