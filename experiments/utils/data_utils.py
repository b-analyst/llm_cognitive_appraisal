"""
Data preprocessing utilities for binary OVA (One-vs-All) emotion probes.

This module contains the BinaryOvaDatasetProcessor class for processing
and managing binary OVA datasets with train/validation splits.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BinaryOvaDatasetProcessor:
    """
    Class for processing and managing binary OVA (One-vs-All) datasets with train/validation splits.
    
    This class handles:
    - Processing raw data into binary OVA format for each emotion
    - Creating train/validation splits
    - Saving processed datasets to output directory
    - Loading processed datasets for training
    
    Example usage:
        ```python
        from experiments.utils.data_utils import BinaryOvaDatasetProcessor
        
        # Initialize processor
        processor = BinaryOvaDatasetProcessor(
            output_dir='outputs/binary_ova_datasets',
            emotions_list=['anger', 'boredom', 'disgust', 'fear', 'guilt', 'joy', 
                          'pride', 'relief', 'sadness', 'shame', 'surprise', 'trust']
        )
        
        # Process data (automatically creates train/val split)
        processor.process_datasets(
            data=train_df,
            emotion_column='emotion',
            text_column='hidden_emo_text',
            val_split=0.15  # 15% validation split
        )
        
        # Load processed datasets for training
        train_labels, train_texts, val_labels, val_texts = processor.load_datasets()
        ```
    """
    
    def __init__(self, output_dir, emotions_list, logger=None):
        """
        Initialize the dataset processor.
        
        Args:
            output_dir: Directory to save processed datasets
            emotions_list: List of emotion names to process
            logger: Optional logger instance
        """
        self.output_dir = output_dir
        self.emotions_list = emotions_list
        self.logger = logger
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"Initialized BinaryOvaDatasetProcessor with output_dir: {output_dir}")
    
    def process_datasets(self, data, emotion_column='emotion', 
                        text_column='hidden_emo_text', filter_valid=True,
                        balance_ratio=1.0, random_state=42, even_negative_distribution=True,
                        val_split=0.15):
        """
        Process data into binary OVA datasets with automatic train/validation split.
        
        For each emotion, creates:
        - Training dataset: {emotion}.csv (85% of data by default)
        - Validation dataset: {emotion}_val.csv (15% of data by default)
        
        The split is done per emotion, maintaining class balance in both splits.
        
        Args:
            data: DataFrame with emotion labels and text
            emotion_column: Name of column containing emotion labels
            text_column: Name of column containing text
            filter_valid: If True, filter out rows with invalid/unrecognized emotions
            balance_ratio: Ratio of negative to positive samples
            random_state: Random state for sampling and splitting
            even_negative_distribution: If True, evenly distribute negative samples
            val_split: Fraction of data to use for validation (default: 0.15)
        
        Returns:
            Dictionary with processing summary
        """
        if self.logger:
            self.logger.info(f"Processing datasets with {val_split*100:.1f}% validation split...")
        
        if emotion_column not in data.columns:
            raise ValueError(f"Column '{emotion_column}' not found in data")
        
        if text_column not in data.columns:
            raise ValueError(f"Column '{text_column}' not found in data")
        
        # Filter valid emotions if requested
        if filter_valid:
            valid_mask = data[emotion_column].isin(self.emotions_list)
            n_filtered = (~valid_mask).sum()
            
            if n_filtered > 0:
                if self.logger:
                    self.logger.warning(f"Filtering out {n_filtered} rows with unrecognized emotions")
                filtered_data = data[valid_mask].copy()
            else:
                filtered_data = data.copy()
        else:
            filtered_data = data.copy()
        
        train_summary = []
        val_summary = []
        
        for emotion in self.emotions_list:
            # Get positive examples (this emotion)
            positive_mask = filtered_data[emotion_column] == emotion
            positive_data = filtered_data[positive_mask].copy()
            n_positive = len(positive_data)
            
            if n_positive == 0:
                if self.logger:
                    self.logger.warning(f"No samples found for '{emotion}', skipping...")
                continue
            
            # Get negative examples (all other emotions)
            negative_mask = filtered_data[emotion_column] != emotion
            negative_data = filtered_data[negative_mask].copy()
            n_negative = len(negative_data)
            
            if n_negative == 0:
                if self.logger:
                    self.logger.warning(f"No negative samples found for '{emotion}', skipping...")
                continue
            
            # Split positive examples into train/val
            pos_train, pos_val = train_test_split(
                positive_data,
                test_size=val_split,
                random_state=random_state,
                shuffle=True
            )
            
            # Split negative examples into train/val
            neg_train, neg_val = train_test_split(
                negative_data,
                test_size=val_split,
                random_state=random_state + 1,  # Different seed for negatives
                shuffle=True
            )
            
            # Process training split
            train_result = self._process_single_emotion(
                positive_data=pos_train,
                negative_data=neg_train,
                emotion=emotion,
                split_name='train',
                emotion_column=emotion_column,
                text_column=text_column,
                balance_ratio=balance_ratio,
                random_state=random_state,
                even_negative_distribution=even_negative_distribution
            )
            if train_result:
                train_summary.append(train_result)
            
            # Process validation split
            val_result = self._process_single_emotion(
                positive_data=pos_val,
                negative_data=neg_val,
                emotion=emotion,
                split_name='val',
                emotion_column=emotion_column,
                text_column=text_column,
                balance_ratio=balance_ratio,
                random_state=random_state + 1000,  # Different seed for validation
                even_negative_distribution=even_negative_distribution
            )
            if val_result:
                val_summary.append(val_result)
        
        train_summary_df = pd.DataFrame(train_summary) if train_summary else pd.DataFrame()
        val_summary_df = pd.DataFrame(val_summary) if val_summary else pd.DataFrame()
        
        if self.logger:
            self.logger.info(f"Processed {len(train_summary)} train datasets")
            self.logger.info(f"Processed {len(val_summary)} validation datasets")
        
        return {
            'train_summary': train_summary_df,
            'val_summary': val_summary_df,
            'output_dir': self.output_dir
        }

    def process_pre_split_datasets(
        self,
        train_data,
        val_data,
        emotion_column='emotion',
        text_column='hidden_emo_text',
        filter_valid=True,
        balance_ratio=1.0,
        random_state=42,
        even_negative_distribution=True,
    ):
        """
        Process already-split train/validation data into binary OVA datasets.

        Use this when the caller has already created leakage-safe scenario-level splits.
        """
        train_result = self._process_single_split(
            train_data,
            split_name='train',
            emotion_column=emotion_column,
            text_column=text_column,
            filter_valid=filter_valid,
            balance_ratio=balance_ratio,
            random_state=random_state,
            even_negative_distribution=even_negative_distribution,
        )
        val_result = self._process_single_split(
            val_data,
            split_name='val',
            emotion_column=emotion_column,
            text_column=text_column,
            filter_valid=filter_valid,
            balance_ratio=balance_ratio,
            random_state=random_state + 1000,
            even_negative_distribution=even_negative_distribution,
        )
        train_summary_df = train_result.get('summary', pd.DataFrame())
        val_summary_df = val_result.get('summary', pd.DataFrame())
        return {
            'train_summary': train_summary_df,
            'val_summary': val_summary_df,
            'output_dir': self.output_dir,
        }
    
    def _process_single_emotion(self, positive_data, negative_data, emotion, split_name,
                               emotion_column='emotion', text_column='hidden_emo_text',
                               balance_ratio=1.0, random_state=42, even_negative_distribution=True):
        """
        Process a single emotion's already-split positive and negative data.
        
        Args:
            positive_data: DataFrame with positive examples (already split)
            negative_data: DataFrame with negative examples (already split)
            emotion: Emotion name
            split_name: Name of split ('train' or 'val')
            emotion_column: Name of column containing emotion labels
            text_column: Name of column containing text
            balance_ratio: Ratio of negative to positive samples
            random_state: Random state for sampling
            even_negative_distribution: If True, evenly distribute negative samples
        
        Returns:
            Dictionary with processing summary or None if failed
        """
        n_positive = len(positive_data)
        n_negative = len(negative_data)
        
        if n_positive == 0 or n_negative == 0:
            return None
        
        # Calculate total negative samples needed
        n_negative_to_sample = int(n_positive * balance_ratio)
        
        if even_negative_distribution:
            # Evenly distribute negative samples across all other emotions
            other_emotions = [e for e in self.emotions_list if e != emotion]
            n_other_emotions = len(other_emotions)
            
            if n_other_emotions == 0:
                return None
            
            # Calculate samples per negative emotion
            samples_per_emotion = max(1, n_negative_to_sample // n_other_emotions)
            remainder = n_negative_to_sample % n_other_emotions
            
            negative_samples_list = []
            negative_distribution = {}
            
            for i, other_emotion in enumerate(other_emotions):
                # Get samples for this negative emotion
                other_emotion_data = negative_data[negative_data[emotion_column] == other_emotion]
                n_available = len(other_emotion_data)
                
                # Calculate how many to sample (distribute remainder across first few emotions)
                n_to_sample = samples_per_emotion + (1 if i < remainder else 0)
                n_to_sample = min(n_to_sample, n_available)
                
                if n_to_sample > 0 and n_available > 0:
                    sampled = other_emotion_data.sample(
                        n=n_to_sample,
                        random_state=random_state + i
                    )
                    negative_samples_list.append(sampled)
                    negative_distribution[other_emotion] = n_to_sample
            
            if negative_samples_list:
                negative_data_sampled = pd.concat(negative_samples_list, ignore_index=True)
                n_negative_sampled = len(negative_data_sampled)
            else:
                return None
        else:
            # Random sampling from all negatives
            if n_negative > n_negative_to_sample:
                negative_data_sampled = negative_data.sample(
                    n=min(n_negative_to_sample, n_negative),
                    random_state=random_state
                ).reset_index(drop=True)
                n_negative_sampled = len(negative_data_sampled)
            else:
                negative_data_sampled = negative_data.copy()
                n_negative_sampled = n_negative
        
        # Create binary labels
        positive_data = positive_data.copy()
        negative_data_sampled = negative_data_sampled.copy()
        positive_data['binary_label'] = 1
        negative_data_sampled['binary_label'] = 0
        
        # Combine positive and negative examples
        combined_data = pd.concat([positive_data, negative_data_sampled], ignore_index=True)
        
        # Shuffle the combined dataset
        combined_data = combined_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Select columns to save
        columns_to_save = [text_column, 'binary_label']
        if emotion_column in combined_data.columns:
            columns_to_save.append(emotion_column)
        
        # Save to CSV
        suffix = '_val' if split_name == 'val' else ''
        filename = f"{emotion}{suffix}.csv"
        filepath = os.path.join(self.output_dir, filename)
        combined_data[columns_to_save].to_csv(filepath, index=False)
        
        if self.logger:
            self.logger.info(f"  {emotion} ({split_name}): {n_positive} pos, {n_negative_sampled} neg -> {len(combined_data)} total -> {filename}")
        
        return {
            'emotion': emotion,
            'split': split_name,
            'n_positive': n_positive,
            'n_negative_total': n_negative,
            'n_negative_sampled': n_negative_sampled,
            'n_total': len(combined_data),
            'positive_ratio': n_positive / len(combined_data),
            'filepath': filepath
        }
    
    def _process_single_split(self, data, split_name, emotion_column='emotion',
                             text_column='hidden_emo_text', filter_valid=True,
                             balance_ratio=1.0, random_state=42, even_negative_distribution=True):
        """
        Process a single split (train or val) into binary OVA datasets.
        
        Args:
            data: DataFrame with emotion labels and text
            split_name: Name of split ('train' or 'val')
            emotion_column: Name of column containing emotion labels
            text_column: Name of column containing text
            filter_valid: If True, filter out rows with invalid/unrecognized emotions
            balance_ratio: Ratio of negative to positive samples
            random_state: Random state for sampling
            even_negative_distribution: If True, evenly distribute negative samples
        
        Returns:
            Dictionary with processing summary
        """
        if emotion_column not in data.columns:
            raise ValueError(f"Column '{emotion_column}' not found in {split_name} data")
        
        if text_column not in data.columns:
            raise ValueError(f"Column '{text_column}' not found in {split_name} data")
        
        # Filter valid emotions if requested
        if filter_valid:
            valid_mask = data[emotion_column].isin(self.emotions_list)
            n_filtered = (~valid_mask).sum()
            
            if n_filtered > 0:
                if self.logger:
                    self.logger.warning(f"Filtering out {n_filtered} rows with unrecognized emotions in {split_name} data")
                filtered_data = data[valid_mask].copy()
            else:
                filtered_data = data.copy()
        else:
            filtered_data = data.copy()
        
        saved_files = []
        summary_stats = []
        
        suffix = '_val' if split_name == 'val' else ''
        
        for emotion in self.emotions_list:
            # Get positive examples (this emotion)
            positive_mask = filtered_data[emotion_column] == emotion
            positive_data = filtered_data[positive_mask].copy()
            n_positive = len(positive_data)
            
            if n_positive == 0:
                if self.logger:
                    self.logger.warning(f"No samples found for '{emotion}' in {split_name} data, skipping...")
                continue
            
            # Get negative examples (all other emotions)
            negative_mask = filtered_data[emotion_column] != emotion
            negative_data = filtered_data[negative_mask].copy()
            n_negative = len(negative_data)
            
            if n_negative == 0:
                if self.logger:
                    self.logger.warning(f"No negative samples found for '{emotion}' in {split_name} data, skipping...")
                continue
            
            # Calculate total negative samples needed
            n_negative_to_sample = int(n_positive * balance_ratio)
            
            if even_negative_distribution:
                # Evenly distribute negative samples across all other emotions
                other_emotions = [e for e in self.emotions_list if e != emotion]
                n_other_emotions = len(other_emotions)
                
                if n_other_emotions == 0:
                    if self.logger:
                        self.logger.warning(f"No other emotions found for '{emotion}' in {split_name} data, skipping...")
                    continue
                
                # Calculate samples per negative emotion
                samples_per_emotion = max(1, n_negative_to_sample // n_other_emotions)
                remainder = n_negative_to_sample % n_other_emotions
                
                negative_samples_list = []
                negative_distribution = {}
                
                for i, other_emotion in enumerate(other_emotions):
                    # Get samples for this negative emotion
                    other_emotion_data = negative_data[negative_data[emotion_column] == other_emotion]
                    n_available = len(other_emotion_data)
                    
                    # Calculate how many to sample (distribute remainder across first few emotions)
                    n_to_sample = samples_per_emotion + (1 if i < remainder else 0)
                    n_to_sample = min(n_to_sample, n_available)
                    
                    if n_to_sample > 0 and n_available > 0:
                        sampled = other_emotion_data.sample(
                            n=n_to_sample,
                            random_state=random_state + i
                        )
                        negative_samples_list.append(sampled)
                        negative_distribution[other_emotion] = n_to_sample
                
                if negative_samples_list:
                    negative_data = pd.concat(negative_samples_list, ignore_index=True)
                    n_negative_sampled = len(negative_data)
                else:
                    if self.logger:
                        self.logger.warning(f"Could not sample enough negative examples for '{emotion}' in {split_name} data, skipping...")
                    continue
            else:
                # Random sampling from all negatives
                if n_negative > n_negative_to_sample:
                    negative_data = negative_data.sample(
                        n=min(n_negative_to_sample, n_negative),
                        random_state=random_state
                    ).reset_index(drop=True)
                    n_negative_sampled = len(negative_data)
                else:
                    n_negative_sampled = n_negative
            
            # Create binary labels
            positive_data['binary_label'] = 1
            negative_data['binary_label'] = 0
            
            # Combine positive and negative examples
            combined_data = pd.concat([positive_data, negative_data], ignore_index=True)
            
            # Shuffle the combined dataset
            combined_data = combined_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # Select columns to save
            columns_to_save = [text_column, 'binary_label']
            if emotion_column in combined_data.columns:
                columns_to_save.append(emotion_column)
            
            # Save to CSV
            filename = f"{emotion}{suffix}.csv"
            filepath = os.path.join(self.output_dir, filename)
            combined_data[columns_to_save].to_csv(filepath, index=False)
            saved_files.append(filepath)
            
            # Store statistics
            summary_stats.append({
                'emotion': emotion,
                'split': split_name,
                'n_positive': n_positive,
                'n_negative_total': n_negative,
                'n_negative_sampled': n_negative_sampled,
                'n_total': len(combined_data),
                'positive_ratio': n_positive / len(combined_data),
                'filepath': filepath
            })
            
            if self.logger:
                self.logger.info(f"  {emotion} ({split_name}): {n_positive} pos, {n_negative_sampled} neg -> {len(combined_data)} total -> {filename}")
        
        summary_df = pd.DataFrame(summary_stats)
        
        return {
            'saved_files': saved_files,
            'summary': summary_df
        }
    
    def load_datasets(self, text_column='hidden_emo_text'):
        """
        Load processed train and validation datasets.
        
        Args:
            text_column: Name of text column in datasets
        
        Returns:
            Tuple of (train_labels_dict, train_texts_dict, val_labels_dict, val_texts_dict)
        """
        train_labels_dict = {}
        train_texts_dict = {}
        val_labels_dict = {}
        val_texts_dict = {}
        
        for emotion in self.emotions_list:
            # Load training dataset
            train_filepath = os.path.join(self.output_dir, f"{emotion}.csv")
            if os.path.exists(train_filepath):
                train_df = pd.read_csv(train_filepath)
                
                if 'binary_label' not in train_df.columns:
                    if self.logger:
                        self.logger.warning(f"'binary_label' column not found in {emotion} training dataset")
                    continue
                
                train_labels_dict[emotion] = train_df['binary_label'].values
                if text_column in train_df.columns:
                    train_texts_dict[emotion] = train_df[text_column].values
            else:
                if self.logger:
                    self.logger.warning(f"Training dataset not found for {emotion}: {train_filepath}")
            
            # Load validation dataset
            val_filepath = os.path.join(self.output_dir, f"{emotion}_val.csv")
            if os.path.exists(val_filepath):
                val_df = pd.read_csv(val_filepath)
                
                if 'binary_label' not in val_df.columns:
                    if self.logger:
                        self.logger.warning(f"'binary_label' column not found in {emotion} validation dataset")
                    continue
                
                val_labels_dict[emotion] = val_df['binary_label'].values
                if text_column in val_df.columns:
                    val_texts_dict[emotion] = val_df[text_column].values
            else:
                if self.logger:
                    self.logger.warning(f"Validation dataset not found for {emotion}: {val_filepath}")
        
        if self.logger:
            self.logger.info(f"Loaded training datasets for {len(train_labels_dict)} emotions")
            self.logger.info(f"Loaded validation datasets for {len(val_labels_dict)} emotions")
        
        return train_labels_dict, train_texts_dict, val_labels_dict, val_texts_dict
