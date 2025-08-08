# fsl-1 main.py - Enhanced with robust labeled test set analysis support
import os
import torch
import sys
import traceback
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader
from datetime import datetime

# Add the directory containing the preprocessing script to the Python path
preprocessing_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(preprocessing_dir)

from config import (
    AUDIO_DIR,
    SPECTROGRAM_DIR,
    EVALUATEAUDIO_DIR,
    EVALUATEDATAPATH,
    BATCH_SIZE,
    DEVICE,
    N_SUPPORT,
    N_QUERY,
    TEST_SIZE,
    REQUIRED_CLASSES,
    N_WAY,
    EPISODES,
    PROTO_WEIGHT,
    RELATION_WEIGHT,
    LABEL_MAP,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS
)

# Import preprocessing and training functions
from src.preprocess import (
    getmetadata,
    create_all_spectrograms,
    check_class_distribution,
    verify_few_shot_requirements,
    getexperimentdata,
    process_audio_file
)
from src.dataset import BirdSoundDataset, SegmentDataset, EpisodicDataLoader
from src.models import CNNEncoder, RelationNetwork, EnsembleModel
from src.training import train_few_shot
from src.evaluation import (
    evaluate_episodic, 
    update_metadata_results,
    evaluate_ensemble_classification,
    evaluate_labeled_test_set,
    load_labeled_test_data,
    LabeledTestDataset,
    update_segment_class_counts,
    update_segment_class_counts_with_time_aggregation,
    create_time_aggregated_summary,
    filter_unprocessed_segments,
    check_if_file_needs_evaluation,
    print_evaluation_results,
    save_evaluation_results,
    ALL_CLASSES,
    NUM_CLASSES
)

# Configuration flags - SET THESE TO CONTROL EVALUATION MODES
ENABLE_TIME_AGGREGATION = False  # Set to True to enable time aggregation
ENABLE_LABELED_TEST = True  # Set to True to enable labeled test set evaluation
LABELED_TEST_ONLY = True  # Set to True to ONLY run labeled test evaluation (skip unlabeled segments)

# Labeled test evaluation will automatically find test files in test/ directory structure:
# Expected structure: test/class_name/audio_files.wav
# Training files should be in: train/class_name/ or validation/class_name/

# When LABELED_TEST_ONLY = True:
# - Only labeled test set evaluation will run
# - Unlabeled segment processing will be skipped
# - Faster execution for testing model performance on labeled data


def split_train_test_data(metadata_df):
    """
    Split metadata into training and test sets based on directory structure.
    Test files should be in test/[class_name]/ folders.
    Training files should be in train/[class_name]/ folders.
    """
    # Filter for test files in test/ directory with proper class structure
    test_metadata = metadata_df[
        metadata_df['file_path'].str.contains('test/') & 
        metadata_df['label'].isin(REQUIRED_CLASSES)
    ]
    
    # Filter for training files in train/ directory (or validation/ if you have it)
    train_metadata = metadata_df[
        (metadata_df['file_path'].str.contains('train/') | 
         metadata_df['file_path'].str.contains('validation/')) & 
        metadata_df['label'].isin(REQUIRED_CLASSES)
    ]
    
    if len(test_metadata) == 0:
        print("WARNING: No test files found in test/ directory!")
        print("Expected structure:")
        for class_name in REQUIRED_CLASSES:
            print(f"  test/{class_name}/")
        print("\nAvailable file paths sample:")
        sample_paths = metadata_df['file_path'].head(10).tolist()
        for path in sample_paths:
            print(f"  {path}")
        return train_metadata, pd.DataFrame()
    
    if len(train_metadata) == 0:
        print("WARNING: No training files found in train/ directory!")
        print("Expected structure:")
        for class_name in REQUIRED_CLASSES:
            print(f"  train/{class_name}/")
        return pd.DataFrame(), test_metadata
    
    print(f"\nDataset split summary:")
    print(f"Training files: {len(train_metadata)} samples")
    print(f"Test files: {len(test_metadata)} samples")
    
    # Show class distribution for training
    train_dist = train_metadata['label'].value_counts()
    print(f"\nTraining set class distribution:")
    for class_name in REQUIRED_CLASSES:
        count = train_dist.get(class_name, 0)
        print(f"  {class_name}: {count} samples")
    
    # Show class distribution for test
    test_dist = test_metadata['label'].value_counts()
    print(f"\nTest set class distribution:")
    for class_name in REQUIRED_CLASSES:
        count = test_dist.get(class_name, 0)
        print(f"  {class_name}: {count} samples")
    
    return train_metadata, test_metadata


def evaluate_on_labeled_test_set(ensemble_model, test_metadata, train_metadata):
    """
    Evaluate the trained model on labeled test set
    """
    print("\n" + "="*80)
    print("EVALUATING ON LABELED TEST SET")
    print("="*80)
    
    if len(test_metadata) == 0:
        print("No test files found! Skipping labeled test evaluation.")
        return None, None, None, None
    
    print(f"Test set contains {len(test_metadata)} samples")
    
    # Check class distribution in test set
    test_dist = test_metadata['label'].value_counts()
    print(f"Test set class distribution:")
    for class_name in REQUIRED_CLASSES:
        count = test_dist.get(class_name, 0)
        print(f"  {class_name}: {count} samples")
    
    # Create test dataset
    test_dataset = LabeledTestDataset(test_metadata)
    
    # Create support dataset from training data
    support_dataset = BirdSoundDataset(train_metadata)
    
    # Evaluate on labeled test set
    print("\nRunning evaluation...")
    results, overall_accuracy, class_accuracies, metrics = evaluate_labeled_test_set(
        model=ensemble_model,
        test_dataset=test_dataset,
        support_dataset=support_dataset,
        device=DEVICE,
        n_way=NUM_CLASSES,
        n_support=N_SUPPORT,
        batch_size=BATCH_SIZE
    )
    
    # Print detailed results
    print_evaluation_results(
        results=results,
        overall_accuracy=overall_accuracy,
        class_accuracies=class_accuracies,
        metrics=metrics,
        show_details=True,
        max_details=30
    )
    
    # Generate output file paths for saving results
    base_path = os.path.join(os.path.dirname(EVALUATEDATAPATH), "labeled_test_results")
    results_path = f"{base_path}_detailed_results.csv"
    summary_path = f"{base_path}_evaluation_summary.csv"
    metrics_path = f"{base_path}_detailed_metrics.csv"
    confusion_path = f"{base_path}_confusion_matrix.csv"
    
    # Save detailed results CSV with all predictions
    print(f"\nSaving evaluation results...")
    results_df = create_detailed_results_csv(
        results=results,
        output_path=results_path,
        include_summary=True
    )
    
    # Save the original evaluation results format (for compatibility)
    save_evaluation_results(
        results=results,
        overall_accuracy=overall_accuracy,
        class_accuracies=class_accuracies,
        metrics=metrics,
        output_path=metrics_path
    )
    
    # Create a confusion matrix CSV
    create_confusion_matrix_csv(results, confusion_path)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"LABELED TEST EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total test samples: {len(results)}")
    print(f"Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Correct predictions: {sum(1 for r in results if r['correct'])}")
    print(f"Incorrect predictions: {sum(1 for r in results if not r['correct'])}")
    
    print(f"\nOutput files generated:")
    print(f"📊 Detailed results: {results_path}")
    print(f"📈 Evaluation summary: {summary_path}")
    print(f"🔢 Detailed metrics: {metrics_path}")
    print(f"🎯 Confusion matrix: {confusion_path}")
    
    return results, overall_accuracy, class_accuracies, metrics


def create_detailed_results_csv(results, output_path, include_summary=True):
    """
    Create a detailed CSV file with all prediction results and metadata.
    
    Args:
        results: List of prediction results
        output_path: Path to save the detailed results CSV
        include_summary: Whether to include summary statistics
    """
    if not results:
        print("No results to save.")
        return
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add additional metadata columns
    results_df['evaluation_timestamp'] = datetime.now().isoformat()
    results_df['model_config'] = f"{NUM_CLASSES}-way_{N_SUPPORT}-shot"
    results_df['file_basename'] = results_df['file_path'].apply(lambda x: os.path.basename(x))
    
    # Reorder columns for better readability
    column_order = [
        'file_path', 'file_basename', 
        'true_label', 'true_class', 
        'predicted_label', 'predicted_class',
        'confidence', 'correct',
        'evaluation_timestamp', 'model_config'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in results_df.columns]
    other_cols = [col for col in results_df.columns if col not in column_order]
    results_df = results_df[column_order + other_cols]
    
    # Save detailed results
    results_df.to_csv(output_path, index=False)
    print(f"✓ Detailed results saved to: {output_path}")
    
    # Create summary file if requested
    if include_summary:
        summary_path = output_path.replace('.csv', '_summary.csv')
        create_evaluation_summary_csv(results, results_df, summary_path)
    
    return results_df


def create_evaluation_summary_csv(results, results_df, summary_path):
    """
    Create a comprehensive summary CSV with metrics and statistics.
    
    Args:
        results: List of prediction results
        results_df: DataFrame version of results
        summary_path: Path to save summary CSV
    """
    # Calculate summary statistics
    total_predictions = len(results)
    correct_predictions = sum(1 for r in results if r.get('correct', False))
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Per-class statistics
    class_stats = {}
    for class_name in ALL_CLASSES:
        # True positives, false positives, false negatives
        tp = len(results_df[(results_df['true_class'] == class_name) & (results_df['predicted_class'] == class_name)])
        fp = len(results_df[(results_df['true_class'] != class_name) & (results_df['predicted_class'] == class_name)])
        fn = len(results_df[(results_df['true_class'] == class_name) & (results_df['predicted_class'] != class_name)])
        tn = len(results_df[(results_df['true_class'] != class_name) & (results_df['predicted_class'] != class_name)])
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        class_stats[class_name] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'support': tp + fn  # Total true instances of this class
        }
    
    # Create summary rows
    summary_data = []
    
    # Overall summary
    summary_data.append({
        'metric_type': 'overall',
        'class_name': 'ALL_CLASSES',
        'accuracy': overall_accuracy,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': total_predictions - correct_predictions,
        'num_classes': NUM_CLASSES,
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_config': f"{NUM_CLASSES}-way_{N_SUPPORT}-shot"
    })
    
    # Per-class summaries
    for class_name, stats in class_stats.items():
        summary_data.append({
            'metric_type': 'per_class',
            'class_name': class_name,
            'accuracy': stats['accuracy'],
            'precision': stats['precision'],
            'recall': stats['recall'],
            'f1_score': stats['f1_score'],
            'true_positives': stats['true_positives'],
            'false_positives': stats['false_positives'],
            'false_negatives': stats['false_negatives'],
            'true_negatives': stats['true_negatives'],
            'support': stats['support'],
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_config': f"{NUM_CLASSES}-way_{N_SUPPORT}-shot"
        })
    
    # Macro averages
    macro_precision = sum(stats['precision'] for stats in class_stats.values()) / len(class_stats)
    macro_recall = sum(stats['recall'] for stats in class_stats.values()) / len(class_stats)
    macro_f1 = sum(stats['f1_score'] for stats in class_stats.values()) / len(class_stats)
    
    summary_data.append({
        'metric_type': 'macro_average',
        'class_name': 'MACRO_AVG',
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1,
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_config': f"{NUM_CLASSES}-way_{N_SUPPORT}-shot"
    })
    
    # Weighted averages
    total_support = sum(stats['support'] for stats in class_stats.values())
    if total_support > 0:
        weighted_precision = sum(stats['precision'] * stats['support'] for stats in class_stats.values()) / total_support
        weighted_recall = sum(stats['recall'] * stats['support'] for stats in class_stats.values()) / total_support
        weighted_f1 = sum(stats['f1_score'] * stats['support'] for stats in class_stats.values()) / total_support
        
        summary_data.append({
            'metric_type': 'weighted_average',
            'class_name': 'WEIGHTED_AVG',
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1,
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_config': f"{NUM_CLASSES}-way_{N_SUPPORT}-shot"
        })
    
    # Create and save summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Evaluation summary saved to: {summary_path}")
    
    return summary_df


def preprocess_data():
    """
    Run preprocessing steps to prepare the dataset.
    """
    print("Starting preprocessing...")
    
    # Scan for new audio files and update metadata
    metadata_df = getmetadata()
    
    # Create spectrograms for all training files
    create_all_spectrograms()
    
    # Check class distribution
    dist = check_class_distribution(metadata_df)
    print(f"Class distribution ({NUM_CLASSES} classes):")
    for cls, count in dist["class_counts"].items():
        print(f" {cls}: {count} samples ({dist['class_percentages'][cls]:.2f}%)")
    
    return metadata_df


def preprocess_evaluation_data():
    """
    Prepare evaluation data by processing audio files into 1-second segments.
    Creates new entries for new files instead of overwriting existing data.
    """
    print("Preparing evaluation data...")
    
    # Get or create experiment metadata
    experiment_df = getexperimentdata()
    
    # Process any unprocessed files (this will create spectrograms for 1-second segments)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # For any unprocessed files, process them into segments
    unprocessed_files = experiment_df[experiment_df['processed'] == False]
    print(f"Found {len(unprocessed_files)} unprocessed files")
    
    for idx, row in unprocessed_files.iterrows():
        try:
            file_path = row['file_path']
            print(f"Processing {file_path}...")
            _, segment_paths = process_audio_file(file_path, mel_spectrogram)
            
            if segment_paths:
                # Update paths in DataFrame
                experiment_df.at[idx, 'spectrogram_paths'] = ','.join(segment_paths)
                experiment_df.at[idx, 'processed'] = True
                print(f"Created {len(segment_paths)} segments for {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save updated DataFrame
    experiment_df.to_csv(EVALUATEDATAPATH, index=False)
    print(f"Updated evaluation data saved to {EVALUATEDATAPATH}")
    return experiment_df


def run_labeled_test_evaluation(model, all_metadata, device):
    """
    Run labeled test set evaluation by automatically splitting train/test data.
    
    Args:
        model: Trained ensemble model
        all_metadata: Complete metadata DataFrame
        device: Computation device
        
    Returns:
        bool: True if evaluation was run successfully, False otherwise
    """
    if not ENABLE_LABELED_TEST:
        print("Labeled test evaluation is disabled in configuration.")
        return False
    
    try:
        print(f"\n{'='*80}")
        print(f"RUNNING LABELED TEST SET EVALUATION")
        print(f"{'='*80}")
        print(f"Model configuration: {NUM_CLASSES}-way {N_SUPPORT}-shot learning")
        print(f"Available classes: {', '.join(ALL_CLASSES)}")
        
        # Split data into train and test sets based on directory structure
        train_metadata, test_metadata = split_train_test_data(all_metadata)
        
        if len(test_metadata) == 0:
            print("ERROR: No test files found in test/ directory structure!")
            print("Please ensure your test files are organized like:")
            for class_name in REQUIRED_CLASSES:
                print(f"  test/{class_name}/audio_files.wav")
            return False
        
        if len(train_metadata) == 0:
            print("ERROR: No training files found!")
            return False
        
        # Run evaluation
        results, overall_accuracy, class_accuracies, metrics = evaluate_on_labeled_test_set(
            ensemble_model=model,
            test_metadata=test_metadata,
            train_metadata=train_metadata
        )
        
        if results is None:
            print("ERROR: Labeled test evaluation failed!")
            return False
        
        print(f"\n✅ Labeled test set evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR during labeled test evaluation: {e}")
        traceback.print_exc()
        return False


def create_confusion_matrix_csv(results, output_path):
    """
    Create a confusion matrix CSV file from results.
    
    Args:
        results: List of prediction results
        output_path: Path to save confusion matrix CSV
    """
    if not results:
        return
    
    # Get unique classes from results
    true_classes = sorted(set(r['true_class'] for r in results))
    pred_classes = sorted(set(r['predicted_class'] for r in results))
    all_classes = sorted(set(true_classes + pred_classes))
    
    # Initialize confusion matrix
    confusion_data = []
    
    # Create matrix data
    for true_class in all_classes:
        row = {'true_class': true_class}
        for pred_class in all_classes:
            count = sum(1 for r in results 
                       if r['true_class'] == true_class and r['predicted_class'] == pred_class)
            row[f'predicted_{pred_class}'] = count
        
        # Add row totals
        row['total_true'] = sum(1 for r in results if r['true_class'] == true_class)
        confusion_data.append(row)
    
    # Add column totals
    totals_row = {'true_class': 'TOTAL_PREDICTED'}
    for pred_class in all_classes:
        total_pred = sum(1 for r in results if r['predicted_class'] == pred_class)
        totals_row[f'predicted_{pred_class}'] = total_pred
    totals_row['total_true'] = len(results)
    confusion_data.append(totals_row)
    
    # Create and save DataFrame
    confusion_df = pd.DataFrame(confusion_data)
    confusion_df.to_csv(output_path, index=False)
    print(f"✓ Confusion matrix saved to: {output_path}")


def main():
    """
    Main function with enhanced support for labeled test set evaluation and CSV output.
    Supports multiple evaluation modes including labeled-test-only mode.
    """
    print(f"Starting FSL-1 evaluation with {NUM_CLASSES} classes: {', '.join(ALL_CLASSES)}")
    
    # Print configuration status
    print(f"\nConfiguration:")
    print(f"  Labeled test evaluation: {'ENABLED' if ENABLE_LABELED_TEST else 'DISABLED'}")
    print(f"  Labeled test ONLY mode: {'ENABLED' if LABELED_TEST_ONLY else 'DISABLED'}")
    if ENABLE_LABELED_TEST:
        print(f"  Test data source: Automatic detection from test/ directory")
        print(f"  Expected structure: test/class_name/audio_files.wav")
    if not LABELED_TEST_ONLY:
        print(f"  Time aggregation: {'ENABLED' if ENABLE_TIME_AGGREGATION else 'DISABLED'}")
        print(f"  Unlabeled segment evaluation: ENABLED")
    else:
        print(f"  Unlabeled segment evaluation: DISABLED (labeled test only mode)")
    
    # Validate labeled test only mode
    if LABELED_TEST_ONLY and not ENABLE_LABELED_TEST:
        print("\nERROR: LABELED_TEST_ONLY is enabled but ENABLE_LABELED_TEST is disabled!")
        print("To use labeled test only mode, you must set both:")
        print("  ENABLE_LABELED_TEST = True")
        print("  LABELED_TEST_ONLY = True")
        return False
    
    # Step 1: Create directories if they don't exist
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    os.makedirs(EVALUATEAUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EVALUATEDATAPATH), exist_ok=True)
    
    try:
        # Step 2: Run preprocessing for training data
        metadata_df = preprocess_data()
        
        # Step 3: Check if we have enough data for few-shot learning
        requirements = verify_few_shot_requirements(
            metadata_df,
            n_way=N_WAY,
            k_shot=N_SUPPORT,
            query_size=N_QUERY,
            test_size=TEST_SIZE
        )
        
        # Step 4: Prepare few-shot experiment
        if requirements["meets_requirements"]:
            # Filter metadata to include only required classes
            all_metadata = metadata_df[metadata_df['label'].isin(REQUIRED_CLASSES)]
            all_dataset = BirdSoundDataset(all_metadata)
            
            # Create support dataset from training data for prototype creation
            train_metadata = all_metadata[~all_metadata['file_path'].str.contains('test/')]
            support_dataset = BirdSoundDataset(train_metadata)
            
            print(f"Training data contains {len(support_dataset)} samples")
            
            # Step 5: Initialize models
            encoder = CNNEncoder().to(DEVICE)
            relation_net = RelationNetwork().to(DEVICE)
            ensemble_model = EnsembleModel(encoder, relation_net).to(DEVICE)
            
            # Step 6: Train the model
            print(f"\nStarting training for {NUM_CLASSES} classes...")
            train_losses = train_few_shot(
                model=ensemble_model,
                dataset=all_dataset,
                episodes=EPISODES,
                n_way=N_WAY,
                n_support=N_SUPPORT,
                n_query=N_QUERY,
                relation_weight=RELATION_WEIGHT,
                proto_weight=PROTO_WEIGHT
            )
            
            print("Training completed!")
            
            # Step 7: Run labeled test set evaluation (if enabled)
            labeled_test_success = run_labeled_test_evaluation(
                model=ensemble_model,
                all_metadata=all_metadata,
                device=DEVICE
            )
            
            # If labeled test only mode, skip unlabeled segment evaluation
            if LABELED_TEST_ONLY:
                print(f"\n{'='*80}")
                print(f"LABELED TEST ONLY MODE - SKIPPING UNLABELED SEGMENT EVALUATION")
                print(f"{'='*80}")
                
                if labeled_test_success:
                    print(f"✅ Labeled test evaluation completed successfully!")
                    print(f"📊 Check output files for detailed results and metrics")
                    return True
                else:
                    print(f"❌ Labeled test evaluation failed!")
                    return False
            
            # Continue with unlabeled segment evaluation (normal mode)
            # Step 8: Prepare evaluation data for unlabeled segments
            print(f"\n{'='*80}")
            # Continue with unlabeled segment evaluation (normal mode)
            # Step 8: Prepare evaluation data for unlabeled segments
            print(f"\n{'='*80}")
            print(f"PREPARING UNLABELED SEGMENT EVALUATION")
            print(f"{'='*80}")
            
            experiment_df = preprocess_evaluation_data()
            
            # Create dataset of all 1-second segments for evaluation
            all_segment_paths = []
            for idx, row in experiment_df.iterrows():
                if row['processed'] and row['spectrogram_paths']:
                    segments = row['spectrogram_paths'].split(',')
                    all_segment_paths.extend(segments)
            
            if not all_segment_paths:
                print("No segments found for evaluation!")
                if labeled_test_success:
                    print("However, labeled test evaluation was completed successfully.")
                    return True
                return False
            
            # FILTER OUT ALREADY PROCESSED SEGMENTS
            filtered_segment_paths, skipped_count = filter_unprocessed_segments(experiment_df, all_segment_paths)
            
            if not filtered_segment_paths:
                print("All files have already been evaluated! No new segments to process.")
                print(f"Total segments: {len(all_segment_paths)}, Already processed: {skipped_count}")
                if labeled_test_success:
                    print("However, labeled test evaluation was completed successfully.")
                    return True
                return False
            
            print(f"Processing {len(filtered_segment_paths)} new segments (skipping {skipped_count} already processed)")
            
            # Create dataset with only unprocessed segments
            segments_df = pd.DataFrame({'file_path': filtered_segment_paths})
            evaluation_dataset = SegmentDataset(segments_df)
            
            # Step 9: Evaluate only the new segments
            print(f"Evaluating model on {len(evaluation_dataset)} NEW segments...")
            results = evaluate_ensemble_classification(
                model=ensemble_model,
                segment_dataset=evaluation_dataset,
                support_dataset=support_dataset,
                device=DEVICE,
                n_way=N_WAY,
                n_support=N_SUPPORT,
                batch_size=BATCH_SIZE
            )

            # Step 10: Update experiment DataFrame with segment class counts
            print("Updating experiment data...")
            if ENABLE_TIME_AGGREGATION:
                print("Using time-based aggregation for metadata update...")
                updated_df = update_segment_class_counts_with_time_aggregation(experiment_df, results)
                
                # Create time aggregated summary if enabled
                try:
                    time_summary = create_time_aggregated_summary(updated_df)
                    if not time_summary.empty:
                        summary_path = EVALUATEDATAPATH.replace('.csv', '_time_summary.csv')
                        time_summary.to_csv(summary_path, index=False)
                        print(f"Time-aggregated summary saved to: {summary_path}")
                except Exception as e:
                    print(f"Warning: Could not create time summary: {e}")
            else:
                print("Using standard file-based counting for metadata update...")
                updated_df = update_metadata_results(results, evaluatedatapath=EVALUATEDATAPATH)
            
            # Step 11: Display final results summary
            print(f"\n{'='*80}")
            print(f"EVALUATION SUMMARY")
            print(f"{'='*80}")
            
            # Count total predictions by class
            class_totals = {}
            for class_name in ALL_CLASSES:
                class_col = f'{class_name}_count'
                if class_col in updated_df.columns:
                    total_count = updated_df[class_col].sum()
                    class_totals[class_name] = total_count
                else:
                    class_totals[class_name] = 0
            
            print(f"Total segments processed: {len(results)}")
            print(f"Files with predictions: {len(updated_df[updated_df[list(class_totals.keys())[0] + '_count'] > 0])}")
            
            print(f"\nClass prediction totals:")
            for class_name, total in class_totals.items():
                print(f"  {class_name}: {total} segments")
            
            print(f"\nResults saved to: {EVALUATEDATAPATH}")
            
            # Final status report
            print(f"\n{'='*80}")
            print(f"FSL-1 EVALUATION COMPLETED")
            print(f"{'='*80}")
            
            if labeled_test_success:
                print(f"✅ Labeled test evaluation: COMPLETED SUCCESSFULLY")
                print(f"✅ Unlabeled segment evaluation: COMPLETED SUCCESSFULLY")
                print(f"\nBoth labeled test files and unlabeled segments have been processed!")
            else:
                print(f"⚠️  Labeled test evaluation: {'SKIPPED (disabled)' if not ENABLE_LABELED_TEST else 'FAILED'}")
                print(f"✅ Unlabeled segment evaluation: COMPLETED SUCCESSFULLY")
            
            return True
                
        else:
            print("\nInsufficient data for few-shot learning requirements:")
            for requirement, status in requirements.items():
                if requirement != "meets_requirements":
                    print(f"  {requirement}: {status}")
            
            print(f"\nRequired: {N_WAY}-way {N_SUPPORT}-shot learning")
            print(f"Need at least {N_SUPPORT + N_QUERY} samples per class for {N_WAY} classes")
            print("Please add more training data or adjust the few-shot parameters in config.py")
            
            return False
    
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()
        
        # Try to provide helpful debugging information
        if 'metadata_df' in locals():
            print(f"\nDebugging info:")
            print(f"- Loaded {len(metadata_df)} training samples")
            if 'all_metadata' in locals():
                print(f"- Filtered to {len(all_metadata)} samples for required classes")
            if 'experiment_df' in locals():
                print(f"- Found {len(experiment_df)} evaluation files")
        
        return False


if __name__ == "__main__":
    success = main()
