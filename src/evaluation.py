# fsl-1 evaluation.py - Updated for flexible labeled test set analysis
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from config import (
    N_WAY, N_SUPPORT, N_QUERY, METADATA_PATH, EPISODES, LEARNING_RATE, 
    PROTO_WEIGHT, RELATION_WEIGHT, LABEL_MAP, EVALUATEDATAPATH, REQUIRED_CLASSES
)

from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np

# Get all classes dynamically from config
ALL_CLASSES = list(LABEL_MAP.keys())
NUM_CLASSES = len(ALL_CLASSES)

# Configuration flag for time aggregation (disabled by default)
ENABLE_TIME_AGGREGATION = False

# Configuration for labeled test set evaluation
LABELED_TEST_ENABLED = False  # Set to True to enable labeled test set evaluation
LABELED_TEST_PATH = None  # Path to labeled test metadata CSV


def calculate_metrics(results, class_labels):
    """
    Calculate F1, precision, recall metrics from evaluation results
    
    Args:
        results: List of prediction results with 'true_label' and 'predicted_label'
        class_labels: List of class labels used in the model
        
    Returns:
        metrics_dict: Dictionary containing overall and per-class metrics
    """
    # Extract true and predicted labels
    y_true = [result['true_label'] for result in results]
    y_pred = [result['predicted_label'] for result in results]
    
    # Calculate overall metrics (macro and weighted averages)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=class_labels, zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=class_labels, zero_division=0
    )
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=class_labels, zero_division=0
    )
    
    # Create metrics dictionary
    metrics_dict = {
        'overall': {
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        },
        'per_class': {}
    }
    
    # Add per-class metrics
    reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
    for i, class_label in enumerate(class_labels):
        class_name = reverse_label_map.get(class_label, f'class_{class_label}')
        metrics_dict['per_class'][class_name] = {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1': f1_per_class[i],
            'support': support_per_class[i]
        }
    
    # Generate classification report for detailed analysis
    class_names = [reverse_label_map.get(label, f'class_{label}') for label in class_labels]
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 labels=class_labels, zero_division=0, output_dict=True)
    metrics_dict['classification_report'] = report
    
    return metrics_dict


def evaluate_labeled_test_set(model, test_dataset, support_dataset, device, n_way=None, n_support=N_SUPPORT, batch_size=32):
    """
    Evaluate labeled test set files and calculate accuracy + additional metrics
    
    Args:
        model: The ensemble model
        test_dataset: Dataset containing labeled test samples  
        support_dataset: Dataset for creating support prototypes
        device: Computation device
        n_way: Number of classes (uses NUM_CLASSES from config if None)
        n_support: Number of support examples per class
        batch_size: Batch size for evaluation
        
    Returns:
        results: List of prediction results with true/false indicators
        accuracy: Overall accuracy
        class_accuracies: Per-class accuracy statistics
        metrics: Dictionary with F1, precision, recall metrics
    """
    # Use dynamic class count from config
    if n_way is None:
        n_way = NUM_CLASSES
    
    model.eval()
    
    print(f"Preparing support set for labeled test evaluation with {n_way} classes...")
    print(f"Available classes: {ALL_CLASSES}")
    
    # Verify test data structure first
    if hasattr(test_dataset, 'data'):
        if not verify_test_data_structure(test_dataset.data):
            print("Test data structure verification failed, but continuing...")
    
    # Prepare support set from training data
    support_data_by_class = {}
    for i, (spectrogram, label) in enumerate(support_dataset):
        if label not in support_data_by_class:
            support_data_by_class[label] = []
        if len(support_data_by_class[label]) < n_support:
            support_data_by_class[label].append(spectrogram)
    
    if len(support_data_by_class) < n_way:
        print(f"Warning: Support dataset has only {len(support_data_by_class)} classes, adjusting n_way to {len(support_data_by_class)}")
        n_way = len(support_data_by_class)
    
    # Use all available classes up to n_way
    class_labels = sorted(support_data_by_class.keys())[:n_way]
    support_images = []
    support_labels = []
    
    class_to_idx = {class_label: idx for idx, class_label in enumerate(class_labels)}
    
    for class_label in class_labels:
        class_spectrograms = support_data_by_class[class_label][:n_support]
        support_images.extend(class_spectrograms)
        support_labels.extend([class_to_idx[class_label]] * len(class_spectrograms))
    
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    print(f"Support set classes: {class_labels}")
    print(f"Support labels mapping: {class_to_idx}")
    
    # Compute prototypes for prototypical network part
    if support_images.dim() == 3:
        support_images = support_images.unsqueeze(1)
    
    with torch.no_grad():
        support_embeddings = model.encoder(support_images, return_embedding=True)
        
        # Compute prototypes for each class
        prototypes = []
        for i in range(n_way):
            class_indices = torch.where(support_labels == i)[0]
            if len(class_indices) > 0:
                class_prototypes = support_embeddings[class_indices].mean(0)
                prototypes.append(class_prototypes)
        prototypes = torch.stack(prototypes)
    
    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_results = []
    correct_predictions = 0
    total_predictions = 0
    class_correct = {i: 0 for i in range(n_way)}
    class_total = {i: 0 for i in range(n_way)}
    
    with torch.no_grad():
        for batch_idx, (batch_spectrograms, batch_true_labels) in enumerate(tqdm(test_loader, desc="Evaluating labeled test set")):
            batch_spectrograms = batch_spectrograms.to(device)
            batch_true_labels = batch_true_labels.to(device)
            
            if batch_spectrograms.dim() == 3:
                batch_spectrograms = batch_spectrograms.unsqueeze(1)
            
            # Get embeddings for query samples
            query_embeddings = model.encoder(batch_spectrograms, return_embedding=True)
            
            for i in range(len(batch_spectrograms)):
                query_embedding = query_embeddings[i]
                true_label = batch_true_labels[i].item()
                
                # Map true label to class index for comparison
                true_class_idx = class_to_idx.get(true_label, -1)
                
                # Prototypical prediction
                dists = torch.cdist(query_embedding.unsqueeze(0), prototypes)
                proto_logits = -dists.squeeze(0)
                proto_probs = F.softmax(proto_logits, dim=0)
            
                
                # Combine predictions
                combined_probs =  proto_probs 
                predicted_idx = torch.argmax(combined_probs).item()
                confidence = combined_probs[predicted_idx].item()
                
                # Determine if prediction is correct
                is_correct = (predicted_idx == true_class_idx)
                if is_correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Update per-class statistics
                if true_class_idx >= 0:
                    class_total[true_class_idx] += 1
                    if is_correct:
                        class_correct[true_class_idx] += 1
                
                # Get class names for display
                reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
                true_class_name = reverse_label_map.get(true_label, 'unknown')
                predicted_class_label = class_labels[predicted_idx] if predicted_idx < len(class_labels) else -1
                predicted_class_name = reverse_label_map.get(predicted_class_label, 'unknown')
                
                # Get file path - use original audio file path for better readability
                sample_idx = batch_idx * batch_size + i
                file_path = test_dataset.get_file_path(sample_idx) if hasattr(test_dataset, 'get_file_path') else f"sample_{sample_idx}"
                
                result = {
                    'file_path': file_path,
                    'true_label': true_label,
                    'true_class': true_class_name,
                    'predicted_label': predicted_class_label,
                    'predicted_class': predicted_class_name,
                    'confidence': confidence,
                    'correct': is_correct
                }
                all_results.append(result)
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Calculate per-class accuracy
    class_accuracies = {}
    reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
    
    for class_idx in range(n_way):
        if class_total[class_idx] > 0:
            class_acc = class_correct[class_idx] / class_total[class_idx]
            class_label = class_labels[class_idx]
            class_name = reverse_label_map.get(class_label, f'class_{class_idx}')
            class_accuracies[class_name] = {
                'accuracy': class_acc,
                'correct': class_correct[class_idx],
                'total': class_total[class_idx]
            }
    
    # Calculate F1, precision, recall metrics
    metrics = calculate_metrics(all_results, class_labels)
    
    return all_results, overall_accuracy, class_accuracies, metrics


def load_labeled_test_data(test_path):
    """
    Load labeled test data from CSV file
    
    Args:
        test_path: Path to labeled test metadata CSV
        
    Returns:
        test_dataset: LabeledTestDataset object
    """
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Labeled test data not found at: {test_path}")
    
    test_metadata = pd.read_csv(test_path)
    print(f"Loaded labeled test data: {len(test_metadata)} samples")
    
    # Verify required columns
    required_cols = ['file_path', 'spectrogram_path', 'label']
    missing_cols = [col for col in required_cols if col not in test_metadata.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in test data: {missing_cols}")
    
    # Check class distribution in test data
    test_class_dist = test_metadata['label'].value_counts()
    print("Test data class distribution:")
    for class_name, count in test_class_dist.items():
        print(f"  {class_name}: {count} samples")
    
    # Create dataset
    test_dataset = LabeledTestDataset(test_metadata)
    return test_dataset


def evaluate_episodic(model, test_dataset, device, n_way=None, n_support=N_SUPPORT, n_query=N_QUERY, n_episodes=EPISODES):
    """
    Evaluate model using episodic few-shot learning paradigm
    
    Args:
        model: Model with encoder and relation_net components
        test_dataset: Dataset for testing
        device: Computation device
        n_way: Number of classes per episode (uses NUM_CLASSES from config if None)
        n_support: Number of support examples per class
        n_query: Number of query examples per class
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Accuracy, detailed results with filenames
    """
    # Use dynamic class count from config
    if n_way is None:
        n_way = NUM_CLASSES
    
    model.eval()
    all_results = []
    
    # Create a DataLoader for batch processing
    data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load metadata for training data to get support samples
    train_metadata = pd.read_csv(METADATA_PATH)
    
    # Create a dictionary to store samples by class
    class_samples = {}
    available_classes = [cls for cls in REQUIRED_CLASSES if cls in train_metadata['label'].values]
    
    for cls in available_classes[:n_way]:  # Limit to n_way classes
        cls_metadata = train_metadata[train_metadata['label'] == cls]
        class_samples[cls] = []
        
        # Load the first n_support samples for each class
        for idx, row in cls_metadata.head(n_support).iterrows():
            try:
                spec_path = row['spectrogram_path']
                if os.path.exists(spec_path):
                    spec = torch.load(spec_path)
                    class_samples[cls].append(spec)
            except Exception as e:
                print(f"Error loading support sample {spec_path}: {e}")
    
    # Convert to tensors and move to device
    support_data = []
    support_labels = []
    
    for cls_idx, cls in enumerate(available_classes[:n_way]):
        for spec in class_samples[cls]:
            if spec.dim() == 2:  # Add channel dimension if needed
                spec = spec.unsqueeze(0)
            support_data.append(spec)
            support_labels.append(cls_idx)
    
    support_data = torch.stack(support_data).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    # Process the support set once to get prototypes
    with torch.no_grad():
        # Get encodings
        support_embeddings = model.encoder(support_data, return_embedding=True)
        
        # Compute prototypes for each class
        prototypes = []
        for i in range(min(n_way, len(available_classes))):
            class_indices = torch.where(support_labels == i)[0]
            if len(class_indices) > 0:
                class_prototypes = support_embeddings[class_indices].mean(0)
                prototypes.append(class_prototypes)
        prototypes = torch.stack(prototypes)
    
    # Process all query samples (evaluation segments)
    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(tqdm(data_loader, desc="Evaluating segments")):
            # Move batch to device
            batch_data = batch_data.to(device)
            if batch_data.dim() == 3:
                batch_data = batch_data.unsqueeze(1)  # Add channel dimension
            
            # Get embeddings
            query_embeddings = model.encoder(batch_data, return_embedding=True)
            
            # Process each embedding in the batch
            for i, query_embedding in enumerate(query_embeddings):
                # Get file path for this sample
                sample_idx = batch_idx * data_loader.batch_size + i
                file_path = test_dataset.get_file_path(sample_idx)
                
                if file_path is None:
                    continue
                
                # Prototypical prediction
                dists = torch.cdist(query_embedding.unsqueeze(0), prototypes)
                proto_logits = -dists.squeeze(0)
                proto_probs = F.softmax(proto_logits, dim=0)
                
                
                # Combine predictions
                combined_probs = PROTO_WEIGHT * proto_probs 
                pred_class = torch.argmax(combined_probs).item()
                confidence = combined_probs[pred_class].item()
                
                # Store result
                result = {
                    'file_path': file_path,
                    'prediction': pred_class,
                    'confidence': confidence,
                }
                all_results.append(result)
    
    return all_results


def evaluate_ensemble_classification(model, segment_dataset, support_dataset, device, n_way=None, n_support=N_SUPPORT, batch_size=32):
    """
    Evaluate segments using the ensemble model (adapted for FSL-1 models)
    Updated to handle dynamic number of classes from config
    """
    # Use dynamic class count from config
    if n_way is None:
        n_way = NUM_CLASSES
    
    model.eval()
    
    print(f"Preparing support set for ensemble evaluation with {n_way} classes...")
    print(f"Available classes: {ALL_CLASSES}")
    
    support_data_by_class = {}
    for i, (spectrogram, label) in enumerate(support_dataset):
        if label not in support_data_by_class:
            support_data_by_class[label] = []
        if len(support_data_by_class[label]) < n_support:
            support_data_by_class[label].append(spectrogram)
    
    if len(support_data_by_class) < n_way:
        print(f"Warning: Support dataset has only {len(support_data_by_class)} classes, adjusting n_way to {len(support_data_by_class)}")
        n_way = len(support_data_by_class)
    
    class_labels = sorted(support_data_by_class.keys())[:n_way]
    support_images = []
    support_labels = []
    
    class_to_idx = {class_label: idx for idx, class_label in enumerate(class_labels)}
    
    for class_label in class_labels:
        class_spectrograms = support_data_by_class[class_label][:n_support]
        support_images.extend(class_spectrograms)
        support_labels.extend([class_to_idx[class_label]] * len(class_spectrograms))
    
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    print(f"Support set classes: {class_labels}")
    print(f"Support labels mapping: {class_to_idx}")
    
    # Compute prototypes for prototypical network part
    if support_images.dim() == 3:
        support_images = support_images.unsqueeze(1)
    
    with torch.no_grad():
        support_embeddings = model.encoder(support_images, return_embedding=True)
        
        # Compute prototypes for each class
        prototypes = []
        for i in range(n_way):
            class_indices = torch.where(support_labels == i)[0]
            if len(class_indices) > 0:
                class_prototypes = support_embeddings[class_indices].mean(0)
                prototypes.append(class_prototypes)
        prototypes = torch.stack(prototypes)
    
    segment_loader = DataLoader(
        segment_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    all_results = []
    segment_idx = 0
    
    with torch.no_grad():
        for batch_spectrograms, _ in tqdm(segment_loader, desc="Evaluating with ensemble"):
            batch_spectrograms = batch_spectrograms.to(device)
            if batch_spectrograms.dim() == 3:
                batch_spectrograms = batch_spectrograms.unsqueeze(1)
            
            # Get embeddings for query samples
            query_embeddings = model.encoder(batch_spectrograms, return_embedding=True)
            
            for i in range(len(batch_spectrograms)):
                current_idx = segment_idx + i
                file_path = segment_dataset.get_file_path(current_idx)
                
                query_embedding = query_embeddings[i]
                
                # Prototypical prediction
                dists = torch.cdist(query_embedding.unsqueeze(0), prototypes)
                proto_logits = -dists.squeeze(0)
                proto_probs = F.softmax(proto_logits, dim=0)
                
                
                # Combine predictions
                combined_probs =  proto_probs 
                predicted_idx = torch.argmax(combined_probs).item()
                confidence = combined_probs[predicted_idx].item()
                
                if predicted_idx < len(class_labels):
                    actual_class_label = class_labels[predicted_idx]
                    reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
                    predicted_label_str = reverse_label_map.get(actual_class_label, 'unknown')
                else:
                    actual_class_label = -1
                    predicted_label_str = 'unknown'
                
                result = {
                    'file_path': file_path,
                    'prediction': predicted_idx,
                    'actual_prediction': actual_class_label,
                    'confidence': confidence,
                    'correct': None
                }
                all_results.append(result)
            
            segment_idx += len(batch_spectrograms)
    
    return all_results


def update_segment_class_counts(experiment_df, results):
    """
    Update segment class counts in experiment DataFrame based on evaluation results.
    This is the default method used when time aggregation is disabled.
    Updated to handle dynamic number of classes from config.
    
    Args:
        experiment_df: DataFrame with experiment files
        results: List of dictionaries with prediction results
        
    Returns:
        experiment_df: Updated DataFrame with class counts
    """
    print(f"Processing {len(results)} results with standard file-based counting for {NUM_CLASSES} classes...")
    
    # Group results by original file path
    file_predictions = {}
    for result in results:
        file_path = result.get('file_path', '')
        if not file_path:
            continue
            
        # Extract the base filename to match with experiment data
        segment_filename = os.path.basename(file_path)
        
        # Extract the original file identifier (without _segXX.pt)
        match = re.search(r'([\w\d]+-[\w\d]+_\d{8}_\d{6})(?:_seg\d+)?\.pt', segment_filename)
        if match:
            original_id = match.group(1)
        else:
            # If pattern doesn't match, try a simpler approach
            original_id = segment_filename.split('_seg')[0]
        
        # Initialize file predictions with all classes from config
        if original_id not in file_predictions:
            file_predictions[original_id] = {class_name: 0 for class_name in ALL_CLASSES}
        
        # Map the prediction to class name using reverse label mapping
        prediction_num = result.get('prediction', -1)
        REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
        prediction = REVERSE_LABEL_MAP.get(prediction_num, "unknown")
        
        # Increment the count for this class
        if prediction in file_predictions[original_id]:
            file_predictions[original_id][prediction] += 1
        else:
            print(f"Warning: Unknown prediction '{prediction}' for file {original_id}")
    
    print(f"Grouped results into {len(file_predictions)} unique files")
    
    # Update counts in experiment DataFrame
    updated_count = 0
    for idx, row in experiment_df.iterrows():
        file_path = row['file_path']
        file_basename = os.path.basename(file_path)
        
        # Extract the identifier part without extension
        original_id = os.path.splitext(file_basename)[0]
        
        if original_id in file_predictions:
            counts = file_predictions[original_id]
            for class_name in ALL_CLASSES:
                experiment_df.at[idx, f'{class_name}_count'] = counts[class_name]
            updated_count += 1
        else:
            # Try alternative matching approaches
            found_match = False
            
            # Try without extension
            base_name_no_ext = os.path.splitext(file_basename)[0]
            if base_name_no_ext in file_predictions:
                counts = file_predictions[base_name_no_ext]
                for class_name in ALL_CLASSES:
                    experiment_df.at[idx, f'{class_name}_count'] = counts[class_name]
                updated_count += 1
                found_match = True
            
            # Try partial matching
            if not found_match:
                for pred_id in file_predictions.keys():
                    if pred_id in base_name_no_ext or base_name_no_ext in pred_id:
                        counts = file_predictions[pred_id]
                        for class_name in ALL_CLASSES:
                            experiment_df.at[idx, f'{class_name}_count'] = counts[class_name]
                        updated_count += 1
                        found_match = True
                        break
            
            if not found_match:
                print(f"Warning: Could not find predictions for file {file_basename}")

    print(f"Updated class counts for {updated_count} files")
    return experiment_df


def update_metadata_results(
    results: List[Dict], 
    test_dataset=None,
    evaluatedatapath=EVALUATEDATAPATH,
) -> pd.DataFrame:
    """
    Updates prediction results in metadata CSV file using string labels.
    Creates new rows for new data instead of overwriting existing data.
    Updated to handle dynamic number of classes from config.
    
    Args:
        results: List of dictionaries containing evaluation results with file_path
        test_dataset: Optional dataset (not required if results contain file paths)
        evaluatedatapath: Path to the evaluation data CSV file
    
    Returns:
        Updated metadata DataFrame
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(evaluatedatapath), exist_ok=True)

    # Read existing metadata file or create new one
    if os.path.exists(evaluatedatapath):
        experiment_df = pd.read_csv(evaluatedatapath)
        print(f"Loaded existing metadata with {len(experiment_df)} rows")
    else:
        print("No existing metadata file found, will create new one")
        experiment_df = pd.DataFrame()

    # Create DataFrame for new results
    new_data = []
    processed_files = set()
    
    for result in results:
        file_path = result.get('file_path', '')
        if not file_path or file_path in processed_files:
            continue
            
        processed_files.add(file_path)
        
        # Check if this file already exists in the metadata
        if not experiment_df.empty and file_path in experiment_df['file_path'].values:
            print(f"File {file_path} already exists in metadata, skipping...")
            continue
        
        # Create new row for this file with all class counts initialized to 0
        new_row = {
            'file_path': file_path,
            'processed': True,
        }
        
        # Initialize all class counts to 0 for all classes from config
        for class_name in ALL_CLASSES:
            new_row[f'{class_name}_count'] = 0
        
        new_data.append(new_row)
    
    # Convert new data to DataFrame
    if new_data:
        new_df = pd.DataFrame(new_data)
        print(f"Created {len(new_df)} new rows for processing")
        
        # Append new data to existing DataFrame
        if experiment_df.empty:
            experiment_df = new_df
        else:
            experiment_df = pd.concat([experiment_df, new_df], ignore_index=True)
    else:
        print("No new files to add to metadata")

    # Choose the appropriate update method based on configuration
    if ENABLE_TIME_AGGREGATION:
        print("Using time-based aggregation for metadata update...")
        updated_df = update_segment_class_counts_with_time_aggregation(experiment_df, results)
    else:
        print("Using standard file-based counting for metadata update...")
        updated_df = update_segment_class_counts(experiment_df, results)

    # Save updated csv (only save once at the end)
    updated_df.to_csv(evaluatedatapath, index=False)

    print(f"Updated prediction results in {evaluatedatapath}")
    return updated_df


def filter_unprocessed_segments(experiment_df, all_segment_paths):
    """
    Filter out segments from files that already have prediction counts.
    Updated to handle dynamic number of classes from config.
    
    Args:
        experiment_df: DataFrame with experiment files and their counts
        all_segment_paths: List of all segment file paths
        
    Returns:
        filtered_segment_paths: List of segments that need evaluation
        already_processed_count: Number of segments skipped
    """
    # Identify files that already have predictions (non-zero or non-null counts)
    processed_files = set()
    
    for idx, row in experiment_df.iterrows():
        # Check if this file already has prediction counts for any class
        has_predictions = False
        total_count = 0
        
        for class_name in ALL_CLASSES:
            count_col = f'{class_name}_count'
            if count_col in row and pd.notna(row[count_col]):
                total_count += row[count_col]
        
        has_predictions = total_count > 0
        
        if has_predictions:
            # Extract the base filename to match against segments
            file_path = row['file_path']
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            processed_files.add(base_filename)
    
    print(f"Found {len(processed_files)} files that already have predictions")
    
    # Filter segment paths to exclude those from already processed files
    filtered_paths = []
    skipped_count = 0
    
    for segment_path in all_segment_paths:
        segment_filename = os.path.basename(segment_path)
        
        # Extract the original file identifier from segment filename
        match = re.search(r'([\w\d]+-[\w\d]+_\d{8}_\d{6})(?:_seg\d+)?\.pt', segment_filename)
        if match:
            original_id = match.group(1)
        else:
            original_id = segment_filename.split('_seg')[0]
        
        if original_id not in processed_files:
            filtered_paths.append(segment_path)
        else:
            skipped_count += 1
    
    print(f"Filtered {len(all_segment_paths)} segments down to {len(filtered_paths)} (skipped {skipped_count})")
    return filtered_paths, skipped_count


def print_evaluation_results(results, overall_accuracy, class_accuracies, metrics=None, show_details=True, max_details=20):
    """
    Print detailed evaluation results including F1, precision, recall
    Updated to handle dynamic number of classes from config.
    
    Args:
        results: List of prediction results
        overall_accuracy: Overall accuracy score
        class_accuracies: Per-class accuracy statistics
        metrics: Dictionary with F1, precision, recall metrics
        show_details: Whether to show individual predictions
        max_details: Maximum number of individual predictions to show
    """
    print(f"\n{'='*60}")
    print(f"LABELED TEST SET EVALUATION RESULTS ({NUM_CLASSES} Classes)")
    print(f"{'='*60}")
    
    print(f"\nAVAILABLE CLASSES: {', '.join(ALL_CLASSES)}")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    if metrics:
        overall_metrics = metrics['overall']
        print(f"Precision (macro): {overall_metrics['precision_macro']:.4f} ({overall_metrics['precision_macro']*100:.2f}%)")
        print(f"Recall (macro): {overall_metrics['recall_macro']:.4f} ({overall_metrics['recall_macro']*100:.2f}%)")
        print(f"F1-score (macro): {overall_metrics['f1_macro']:.4f} ({overall_metrics['f1_macro']*100:.2f}%)")
        print(f"\nWeighted averages:")
        print(f"Precision (weighted): {overall_metrics['precision_weighted']:.4f} ({overall_metrics['precision_weighted']*100:.2f}%)")
        print(f"Recall (weighted): {overall_metrics['recall_weighted']:.4f} ({overall_metrics['recall_weighted']*100:.2f}%)")
        print(f"F1-score (weighted): {overall_metrics['f1_weighted']:.4f} ({overall_metrics['f1_weighted']*100:.2f}%)")
    
    print(f"\nTotal predictions: {len(results)}")
    correct_count = sum(1 for r in results if r['correct'])
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {len(results) - correct_count}")
    
    print(f"\nPER-CLASS PERFORMANCE:")
    if metrics and 'per_class' in metrics:
        print(f"{'Class':<15} {'Acc %':<8} {'Prec %':<8} {'Rec %':<8} {'F1 %':<8} {'Support':<8}")
        print(f"{'-'*65}")
        for class_name, stats in class_accuracies.items():
            acc_pct = stats['accuracy'] * 100
            
            # Get metrics for this class
            class_metrics = metrics['per_class'].get(class_name, {})
            prec_pct = class_metrics.get('precision', 0) * 100
            rec_pct = class_metrics.get('recall', 0) * 100
            f1_pct = class_metrics.get('f1', 0) * 100
            support = class_metrics.get('support', stats['total'])
            
            print(f"{class_name:<15} {acc_pct:>6.1f}  {prec_pct:>6.1f}  {rec_pct:>6.1f}  {f1_pct:>6.1f}  {support:>6}")
    else:
        print(f"{'Class':<15} {'Accuracy':<10} {'Correct/Total':<15}")
        print(f"{'-'*40}")
        for class_name, stats in class_accuracies.items():
            acc_pct = stats['accuracy'] * 100
            print(f"{class_name:<15} {acc_pct:>7.2f}%   {stats['correct']:>3}/{stats['total']:<3}")
    
    if show_details:
        print(f"\nDETAILED PREDICTIONS (showing first {max_details}):")
        print(f"{'File':<30} {'True':<15} {'Predicted':<15} {'Correct':<8} {'Confidence':<10}")
        print(f"{'-'*85}")
        
        for i, result in enumerate(results[:max_details]):
            file_name = os.path.basename(result['file_path'])[:28]
            true_class = result['true_class'][:13]
            pred_class = result['predicted_class'][:13]
            correct_str = "✓" if result['correct'] else "✗"
            confidence = result['confidence']
            
            print(f"{file_name:<30} {true_class:<15} {pred_class:<15} {correct_str:<8} {confidence:>8.3f}")
        
        if len(results) > max_details:
            print(f"... and {len(results) - max_details} more predictions")
    
    # Show confusion-like statistics
    print(f"\nPREDICTION BREAKDOWN:")
    pred_matrix = {}
    for result in results:
        true_class = result['true_class']
        pred_class = result['predicted_class']
        
        if true_class not in pred_matrix:
            pred_matrix[true_class] = {}
        if pred_class not in pred_matrix[true_class]:
            pred_matrix[true_class][pred_class] = 0
        pred_matrix[true_class][pred_class] += 1
    
    for true_class, predictions in pred_matrix.items():
        print(f"\nTrue class '{true_class}':")
        for pred_class, count in predictions.items():
            status = "✓" if true_class == pred_class else "✗"
            print(f"  {status} Predicted as '{pred_class}': {count}")


def save_evaluation_results(results, overall_accuracy, class_accuracies, metrics=None, output_path=None):
    """
    Save evaluation results to CSV file with F1, precision, recall metrics
    Updated to handle dynamic number of classes from config.
    
    Args:
        results: List of prediction results
        overall_accuracy: Overall accuracy score
        class_accuracies: Per-class accuracy statistics
        metrics: Dictionary with F1, precision, recall metrics
        output_path: Path to save results (optional)
    """
    if output_path is None:
        output_path = EVALUATEDATAPATH
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add summary statistics as metadata
    summary_data = {
        'overall_accuracy': overall_accuracy,
        'total_predictions': len(results),
        'correct_predictions': sum(1 for r in results if r['correct']),
        'num_classes': NUM_CLASSES,
        'classes': ','.join(ALL_CLASSES),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Add overall metrics if available
    if metrics and 'overall' in metrics:
        overall_metrics = metrics['overall']
        summary_data.update({
            'precision_macro': overall_metrics['precision_macro'],
            'recall_macro': overall_metrics['recall_macro'],
            'f1_macro': overall_metrics['f1_macro'],
            'precision_weighted': overall_metrics['precision_weighted'],
            'recall_weighted': overall_metrics['recall_weighted'],
            'f1_weighted': overall_metrics['f1_weighted']
        })
    
    # Add per-class accuracies and metrics to summary
    for class_name, stats in class_accuracies.items():
        summary_data[f'{class_name}_accuracy'] = stats['accuracy']
        summary_data[f'{class_name}_correct'] = stats['correct']
        summary_data[f'{class_name}_total'] = stats['total']
        
        # Add per-class metrics if available
        if metrics and 'per_class' in metrics and class_name in metrics['per_class']:
            class_metrics = metrics['per_class'][class_name]
            summary_data[f'{class_name}_precision'] = class_metrics['precision']
            summary_data[f'{class_name}_recall'] = class_metrics['recall']
            summary_data[f'{class_name}_f1'] = class_metrics['f1']
    
    # Save results
    results_df.to_csv(output_path, index=False)
    
    # Save summary separately
    summary_path = output_path.replace('.csv', '_summary.csv')
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    
    # Also save detailed metrics if available
    if metrics and 'classification_report' in metrics:
        metrics_path = output_path.replace('.csv', '_detailed_metrics.csv')
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        report_df.to_csv(metrics_path)
        print(f"Detailed metrics saved to: {metrics_path}")


class LabeledTestDataset(torch.utils.data.Dataset):
    """Dataset class for labeled test data that can return file paths"""
    
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        # Store both spectrogram paths and original file paths
        self.file_paths = list(dataframe['file_path'])  # Original audio file paths
        self.spectrogram_paths = list(dataframe['spectrogram_path'])  # Spectrogram paths
        self.labels = [LABEL_MAP[label] for label in dataframe['label']]  # Convert to numeric
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spectrogram_path = self.spectrogram_paths[idx]
        label = self.labels[idx]
        
        # Load spectrogram
        spectrogram = torch.load(spectrogram_path)
        
        # Apply transform if provided
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        # Ensure proper dimensions for CNN (add channel dimension if needed)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # Add channel dimension
        
        return spectrogram, label
    
    def get_file_path(self, idx):
        """Get original audio file path for given index"""
        if idx < len(self.file_paths):
            return self.file_paths[idx]
        return None
    
    def get_spectrogram_path(self, idx):
        """Get spectrogram file path for given index"""
        if idx < len(self.spectrogram_paths):
            return self.spectrogram_paths[idx]
        return None


def verify_test_data_structure(test_metadata):
    """
    Verify that test data has the expected directory structure and files
    Updated to handle dynamic classes from config.
    """
    print("Verifying test data structure...")
    
    # Create expected directories based on config classes
    required_dirs = [f'test/{class_name}' for class_name in ALL_CLASSES]
    found_dirs = set()
    
    for _, row in test_metadata.iterrows():
        file_path = row['file_path']
        for req_dir in required_dirs:
            if req_dir in file_path:
                found_dirs.add(req_dir)
    
    print(f"Found test directories: {list(found_dirs)}")
    missing_dirs = [d for d in required_dirs if d not in found_dirs]
    
    if missing_dirs:
        print(f"WARNING: Missing test directories: {missing_dirs}")
        return False
    
    # Check if spectrograms exist
    missing_spectrograms = 0
    for _, row in test_metadata.iterrows():
        spec_path = row['spectrogram_path']
        if not os.path.exists(spec_path):
            missing_spectrograms += 1
    
    if missing_spectrograms > 0:
        print(f"WARNING: {missing_spectrograms} test spectrograms are missing!")
        print("Run preprocessing to create missing spectrograms.")
        return False
    
    print("✓ Test data structure verification passed")
    return True


def check_if_file_needs_evaluation(experiment_df, file_path):
    """
    Check if a specific file needs evaluation based on existing prediction counts.
    Updated to handle dynamic number of classes from config.
    
    Args:
        experiment_df: DataFrame with experiment data
        file_path: Path to the file to check
        
    Returns:
        bool: True if file needs evaluation, False if already processed
    """
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Find matching row in experiment_df
    matching_rows = experiment_df[
        experiment_df['file_path'].str.contains(base_filename, na=False)
    ]
    
    if len(matching_rows) == 0:
        return True  # New file, needs evaluation
    
    row = matching_rows.iloc[0]
    
    # Check if predictions already exist for all classes
    class_counts = {}
    for class_name in ALL_CLASSES:
        count_col = f'{class_name}_count'
        class_counts[class_name] = row.get(count_col, None)
    
    # Check if all class counts exist and at least one is non-zero
    has_predictions = (
        all(pd.notna(count) for count in class_counts.values()) and
        sum(class_counts.values()) > 0
    )
    
    return not has_predictions


# Time aggregation functions (kept for compatibility but not used by default)
def extract_time_from_filename(filename):
    """Extract time information from filename for time aggregation (optional feature)"""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    patterns = [
        r'(\d{8})_(\d{6})',  # YYYYMMDD_HHMMSS
        r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
        r'(\d{4}\d{2}\d{2})_(\d{2}\d{2}\d{2})',  # YYYYMMDD_HHMMSS
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base_name)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)
            
            if len(date_part) == 8:  # YYYYMMDD
                year = date_part[:4]
                month = date_part[4:6]
                day = date_part[6:8]
                date_str = f"{year}-{month}-{day}"
            else:
                date_str = date_part
            
            if len(time_part) >= 2:
                hour = time_part[:2]
                return f"{date_str}_{hour}"
    
    return base_name


def update_segment_class_counts_with_time_aggregation(experiment_df, results):
    """Time-based aggregation method (optional feature, disabled by default)"""
    print(f"Processing {len(results)} results with time-based aggregation...")
    print("Note: This feature is experimental and disabled by default.")
    
    # Implement time aggregation logic here if needed
    # For now, fall back to standard counting
    return update_segment_class_counts(experiment_df, results)


def create_time_aggregated_summary(experiment_df):
    """Create time-aggregated summary (optional feature, disabled by default)"""
    print("Creating time-aggregated summary (experimental feature)...")
    return pd.DataFrame()  # Return empty for now