import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')


class AdvancedAHUSplitCreator:
    """
    Advanced split creator with multiple strategies for handling extreme class imbalance
    """

    def __init__(self,
                 test_size=0.15,
                 val_size=0.15,
                 random_state=42,
                 min_samples_for_split=20,
                 min_samples_per_class_per_ahu=100,
                 smote_method='smote',
                 smote_ratio='auto',
                 k_neighbors=5,
                 impute_strategy='constant',
                 impute_fill_value=0):
        """
        Parameters:
        - test_size: proportion for test set
        - val_size: proportion for validation set
        - random_state: random seed
        - min_samples_for_split: minimum samples required to perform stratified split
        - min_samples_per_class_per_ahu: threshold for augmenting rare classes (default 100)
        - smote_method: 'smote', 'borderline', 'adasyn', 'smote_tomek', 'smote_enn'
        - smote_ratio: 'auto', 'minority', 'not majority', or dict/float
        - k_neighbors: neighbors for SMOTE
        - impute_strategy: strategy for imputing NaN values ('constant', 'mean', 'median')
        - impute_fill_value: value to use for constant imputation (0 = heating/cooling off)
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.min_samples_for_split = min_samples_for_split
        self.min_samples_per_class_per_ahu = min_samples_per_class_per_ahu
        self.smote_method = smote_method
        self.smote_ratio = smote_ratio
        self.k_neighbors = k_neighbors
        self.impute_strategy = impute_strategy
        self.impute_fill_value = impute_fill_value

    def _impute_nan_values(self, X):
        """
        Impute NaN values in the dataset

        NaN values in AHU data are seasonal:
        - Heating-related columns (Heat Exchanger, Heating supply temperature) = NaN in summer
        - Cooling-related columns (Cooling supply temperature, cooling pump) = NaN in winter

        We fill them with 0 to indicate "off" state
        """
        if X.isnull().sum().sum() == 0:
            return X

        X_copy = X.copy()
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns

        imputer = SimpleImputer(strategy=self.impute_strategy, fill_value=self.impute_fill_value)
        X_copy[numeric_cols] = imputer.fit_transform(X_copy[numeric_cols])

        return X_copy

    def _get_smote_sampler(self, k_neighbors_adjusted):
        """Get SMOTE sampler based on method"""

        if self.smote_method == 'smote':
            return SMOTE(
                sampling_strategy=self.smote_ratio,
                random_state=self.random_state,
                k_neighbors=k_neighbors_adjusted
            )
        elif self.smote_method == 'borderline':
            return BorderlineSMOTE(
                sampling_strategy=self.smote_ratio,
                random_state=self.random_state,
                k_neighbors=k_neighbors_adjusted
            )
        elif self.smote_method == 'adasyn':
            return ADASYN(
                sampling_strategy=self.smote_ratio,
                random_state=self.random_state,
                n_neighbors=k_neighbors_adjusted
            )
        elif self.smote_method == 'smote_tomek':
            smote = SMOTE(
                sampling_strategy=self.smote_ratio,
                random_state=self.random_state,
                k_neighbors=k_neighbors_adjusted
            )
            return SMOTETomek(smote=smote, random_state=self.random_state)
        elif self.smote_method == 'smote_enn':
            smote = SMOTE(
                sampling_strategy=self.smote_ratio,
                random_state=self.random_state,
                k_neighbors=k_neighbors_adjusted
            )
            return SMOTEENN(smote=smote, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown SMOTE method: {self.smote_method}")

    def _safe_stratified_split(self, X, y, test_size, split_name="split"):
        """
        Perform stratified split with fallback to random split if stratification fails
        """
        try:
            # Check if we have enough samples per class
            class_counts = Counter(y)
            min_class_count = min(class_counts.values())

            if min_class_count < 2:
                print(f"Warning: {split_name} - Class with only 1 sample, using random split")
                return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

            # Try stratified split
            return train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=self.random_state
            )
        except ValueError as e:
            print(f"Warning: {split_name} - Stratification failed ({e}), using random split")
            return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

    def _augment_rare_classes(self, splits_dict, df_full, ahu_col='AHU_name', label_col='label'):
        """
        Augment training data for AHUs with rare classes by adding real samples from other AHUs.
        This ensures every AHU client has enough samples of each class for effective learning.
        
        Strategy:
        - For each AHU, check training set class distribution
        - If a class has < min_samples_per_class_per_ahu samples, augment it
        - Add real samples from other AHUs (maintaining federated privacy reasonable limits)
        """
        print(f"\n{'='*80}")
        print("STRATIFIED AHU AUGMENTATION")
        print(f"Target: {self.min_samples_per_class_per_ahu} samples per class per AHU")
        print("="*80)
        
        # Create global pool of samples by class
        global_pools = {}
        all_classes = sorted(df_full[label_col].unique())
        
        for cls in all_classes:
            class_df = df_full[df_full[label_col] == cls]
            X_pool = class_df.drop(columns=[label_col])
            y_pool = class_df[label_col]
            global_pools[cls] = (X_pool, y_pool)
            print(f"  Global pool C{cls}: {len(y_pool)} samples")
        
        # Augment each AHU
        augmented = {}
        for ahu, data in splits_dict['federated'].items():
            X_train, y_train = data['train']
            X_val, y_val = data['val']
            X_test, y_test = data['test']
            
            print(f"\n{ahu}:")
            class_counts = Counter(y_train)
            print(f"  Before: {dict(class_counts)}")
            
            # Check which classes need augmentation
            needs_augmentation = False
            for cls in all_classes:
                count = class_counts.get(cls, 0)
                if count < self.min_samples_per_class_per_ahu:
                    needs_augmentation = True
                    break
            
            if not needs_augmentation:
                print(f"  ✓ All classes have >= {self.min_samples_per_class_per_ahu} samples")
                augmented[ahu] = data
                continue
            
            # Augment rare classes
            X_train_aug = X_train.copy()
            y_train_aug = y_train.copy()
            
            for cls in all_classes:
                current_count = class_counts.get(cls, 0)
                needed = self.min_samples_per_class_per_ahu - current_count
                
                if needed <= 0:
                    continue
                
                # Get samples from global pool (excluding current AHU to maintain some separation)
                X_pool, y_pool = global_pools[cls]
                
                # Try to get samples from other AHUs
                pool_indices = X_pool.index
                available_indices = [idx for idx in pool_indices if idx not in X_train.index]
                
                if len(available_indices) == 0:
                    print(f"    C{cls}: No external samples available")
                    continue
                
                # Sample up to 'needed' samples
                n_samples = min(needed, len(available_indices))
                np.random.seed(self.random_state)
                sampled_indices = np.random.choice(available_indices, size=n_samples, replace=False)
                
                # Add to training set
                X_add = X_pool.loc[sampled_indices]
                y_add = y_pool.loc[sampled_indices]
                
                X_train_aug = pd.concat([X_train_aug, X_add], ignore_index=True)
                y_train_aug = pd.concat([y_train_aug, y_add], ignore_index=True)
                
                print(f"    C{cls}: {current_count} → {current_count + n_samples} (+{n_samples} from other AHUs)")
            
            # Store augmented data
            augmented[ahu] = {
                'train': (X_train_aug, y_train_aug),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
            
            print(f"  After: {dict(Counter(y_train_aug))}")
        
        return augmented

    def create_ahu_splits(self, df, ahu_col='AHU_name', label_col='label'):
        """
        Create stratified splits per AHU with intelligent handling of edge cases
        """
        results = {
            'centralized': {},
            'federated': {},
            'metadata': {
                'ahu_stats': {},
                'excluded_ahus': []
            }
        }

        ahus = sorted(df[ahu_col].unique())

        # Process each AHU
        for ahu in ahus:
            ahu_df = df[df[ahu_col] == ahu].copy()

            print(f"\nProcessing {ahu}:")
            print(f"  Total samples: {len(ahu_df)}")

            # Get class distribution
            class_dist = Counter(ahu_df[label_col])
            print(f"  Class distribution: {dict(class_dist)}")

            # Store metadata
            results['metadata']['ahu_stats'][ahu] = {
                'total_samples': len(ahu_df),
                'class_distribution': dict(class_dist),
                'num_classes': len(class_dist)
            }

            # Check if AHU has sufficient data
            if len(ahu_df) < self.min_samples_for_split:
                print(f"  -> Insufficient samples (< {self.min_samples_for_split}), excluding from federated training")
                results['metadata']['excluded_ahus'].append(ahu)
                continue

            # Check if only one class (e.g., AHU-7 with 0% faults)
            if len(class_dist) == 1:
                print(f"  -> Only one class present, excluding from federated training")
                results['metadata']['excluded_ahus'].append(ahu)
                continue

            X = ahu_df.drop(columns=[label_col])
            y = ahu_df[label_col]

            # Create splits
            X_temp, X_test, y_temp, y_test = self._safe_stratified_split(
                X, y, self.test_size, f"{ahu} train/val vs test"
            )

            val_size_adjusted = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = self._safe_stratified_split(
                X_temp, y_temp, val_size_adjusted, f"{ahu} train vs val"
            )

            results['federated'][ahu] = {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }

            print(f"  -> Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        # Create centralized splits
        print(f"\nProcessing CENTRALIZED (all AHUs):")
        print(f"  Total samples: {len(df)}")

        class_dist = Counter(df[label_col])
        print(f"  Class distribution: {dict(class_dist)}")

        X_all = df.drop(columns=[label_col])
        y_all = df[label_col]

        X_temp, X_test_c, y_temp, y_test_c = self._safe_stratified_split(
            X_all, y_all, self.test_size, "centralized train/val vs test"
        )

        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train_c, X_val_c, y_train_c, y_val_c = self._safe_stratified_split(
            X_temp, y_temp, val_size_adjusted, "centralized train vs val"
        )

        results['centralized'] = {
            'train': (X_train_c, y_train_c),
            'val': (X_val_c, y_val_c),
            'test': (X_test_c, y_test_c)
        }

        print(f"  -> Train: {len(y_train_c)}, Val: {len(y_val_c)}, Test: {len(y_test_c)}")

        # Augment rare classes in federated splits
        if results['federated']:
            augmented_federated = self._augment_rare_classes(results, df, ahu_col, label_col)
            results['federated'] = augmented_federated

        return results

    def apply_smote(self, splits_dict, apply_to_val=False):
        """
        Apply SMOTE to training data (and optionally validation data)

        Parameters:
        - splits_dict: dictionary from create_ahu_splits
        - apply_to_val: whether to apply SMOTE to validation set (not recommended)
        """
        results = {
            'centralized': {},
            'federated': {},
            'metadata': splits_dict.get('metadata', {})
        }

        results['metadata']['smote_applied'] = {
            'method': self.smote_method,
            'ratio': self.smote_ratio,
            'k_neighbors': self.k_neighbors
        }

        # Apply SMOTE to centralized
        if 'centralized' in splits_dict:
            print(f"\nApplying SMOTE to centralized training data...")

            X_train, y_train = splits_dict['centralized']['train']
            X_val, y_val = splits_dict['centralized']['val']
            X_test, y_test = splits_dict['centralized']['test']

            # Get numeric columns
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            X_train_numeric = X_train[numeric_cols]
            X_val_numeric = X_val[numeric_cols]
            X_test_numeric = X_test[numeric_cols]

            # Apply SMOTE to training
            X_train_resampled, y_train_resampled = self._apply_smote_to_data(
                X_train_numeric, y_train, "Centralized"
            )

            # Optionally apply to validation
            if apply_to_val:
                X_val_resampled, y_val_resampled = self._apply_smote_to_data(
                    X_val_numeric, y_val, "Centralized Val"
                )
            else:
                # Just impute val/test without SMOTE
                X_val_resampled = self._impute_nan_values(X_val_numeric)
                y_val_resampled = y_val

            # Impute test
            X_test_resampled = self._impute_nan_values(X_test_numeric)

            results['centralized'] = {
                'train': (X_train_resampled, y_train_resampled),
                'val': (X_val_resampled, y_val_resampled),
                'test': (X_test_resampled, y_test)
            }

        # Apply SMOTE to federated
        if 'federated' in splits_dict:
            for ahu, data in splits_dict['federated'].items():
                print(f"\nApplying SMOTE to {ahu} training data...")

                X_train, y_train = data['train']
                X_val, y_val = data['val']
                X_test, y_test = data['test']

                # Get numeric columns
                numeric_cols = X_train.select_dtypes(include=[np.number]).columns
                X_train_numeric = X_train[numeric_cols]
                X_val_numeric = X_val[numeric_cols]
                X_test_numeric = X_test[numeric_cols]

                # Apply SMOTE to training
                X_train_resampled, y_train_resampled = self._apply_smote_to_data(
                    X_train_numeric, y_train, ahu
                )

                # Optionally apply to validation
                if apply_to_val:
                    X_val_resampled, y_val_resampled = self._apply_smote_to_data(
                        X_val_numeric, y_val, f"{ahu} Val"
                    )
                else:
                    # Just impute val/test without SMOTE
                    X_val_resampled = self._impute_nan_values(X_val_numeric)
                    y_val_resampled = y_val

                # Impute test
                X_test_resampled = self._impute_nan_values(X_test_numeric)

                results['federated'][ahu] = {
                    'train': (X_train_resampled, y_train_resampled),
                    'val': (X_val_resampled, y_val_resampled),
                    'test': (X_test_resampled, y_test)
                }

        return results

    def _apply_smote_to_data(self, X, y, name="Data"):
        """Apply SMOTE to given data"""

        # Impute NaN values first
        X_imputed = self._impute_nan_values(X)

        class_counts = Counter(y)
        min_class_count = min(class_counts.values())

        # Adjust k_neighbors
        k_neighbors_adjusted = min(self.k_neighbors, min_class_count - 1)

        if k_neighbors_adjusted < 1 or len(class_counts) < 2:
            print(f"  {name}: Cannot apply SMOTE (k_neighbors={k_neighbors_adjusted}, classes={len(class_counts)})")
            print(f"  {name}: Returning imputed data")
            return X_imputed, y

        try:
            sampler = self._get_smote_sampler(k_neighbors_adjusted)
            X_resampled, y_resampled = sampler.fit_resample(X_imputed, y)

            # Convert to DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled, columns=X_imputed.columns)
            y_resampled = pd.Series(y_resampled, name='label')

            print(f"  {name}: {len(y)} -> {len(y_resampled)} samples")
            print(f"  {name}: Original class dist: {dict(Counter(y))}")
            print(f"  {name}: After SMOTE class dist: {dict(Counter(y_resampled))}")

            return X_resampled, y_resampled

        except Exception as e:
            print(f"  {name}: SMOTE failed - {e}")
            print(f"  {name}: Returning imputed data")
            return X_imputed, y

    def save_splits(self, splits_dict, output_dir='processed_splits_advanced'):
        """Save splits to files"""

        os.makedirs(output_dir, exist_ok=True)

        # Save centralized
        if 'centralized' in splits_dict:
            cent_dir = os.path.join(output_dir, 'centralized')
            os.makedirs(cent_dir, exist_ok=True)

            for split_name, (X, y) in splits_dict['centralized'].items():
                df_split = X.copy()
                df_split['label'] = y.values

                filepath = os.path.join(cent_dir, f'{split_name}.csv')
                df_split.to_csv(filepath, index=False)
                print(f"\nSaved: {filepath} ({df_split.shape})")

        # Save federated
        if 'federated' in splits_dict:
            fed_dir = os.path.join(output_dir, 'federated')
            os.makedirs(fed_dir, exist_ok=True)

            for ahu, data in splits_dict['federated'].items():
                ahu_dir = os.path.join(fed_dir, str(ahu))
                os.makedirs(ahu_dir, exist_ok=True)

                for split_name, (X, y) in data.items():
                    df_split = X.copy()
                    df_split['label'] = y.values

                    filepath = os.path.join(ahu_dir, f'{split_name}.csv')
                    df_split.to_csv(filepath, index=False)
                    print(f"Saved: {filepath} ({df_split.shape})")

        # Save metadata
        if 'metadata' in splits_dict:
            import json
            metadata_path = os.path.join(output_dir, 'metadata.json')

            # Convert numpy types to native Python types for JSON serialization
            metadata = splits_dict['metadata']
            metadata_serializable = self._make_json_serializable(metadata)

            with open(metadata_path, 'w') as f:
                json.dump(metadata_serializable, f, indent=2)
            print(f"\nSaved: {metadata_path}")

    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            # Convert both keys and values
            return {
                self._make_json_serializable(k): self._make_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def convert_onehot_to_ahu_column(df, ahu_columns=None):
    """
    Convert one-hot encoded AHU columns to a single AHU_name column

    Parameters:
    - df: DataFrame with one-hot encoded ahu_1, ahu_2, ..., ahu_8 columns
    - ahu_columns: list of AHU column names (auto-detected if None)

    Returns:
    - DataFrame with AHU_name column added
    """
    if ahu_columns is None:
        # Auto-detect ahu columns
        ahu_columns = [col for col in df.columns if col.startswith('ahu_')]

    if not ahu_columns:
        raise ValueError("No AHU columns found in the dataset")

    print(f"Found AHU columns: {ahu_columns}")

    # Create AHU_name column from one-hot encoding
    df_copy = df.copy()

    # Find which AHU each row belongs to
    ahu_name = []
    for idx, row in df_copy.iterrows():
        for ahu_col in ahu_columns:
            if row[ahu_col] == 1:
                ahu_name.append(ahu_col)
                break
        else:
            # No AHU found (shouldn't happen)
            ahu_name.append('unknown')

    df_copy['AHU_name'] = ahu_name

    print(f"AHU distribution:")
    ahu_dist = df_copy['AHU_name'].value_counts().sort_index()
    for ahu, count in ahu_dist.items():
        print(f"  {ahu}: {count} samples")

    return df_copy


def main():
    """Main execution with advanced options"""
    
    # Set seed for reproducibility
    import random
    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    print(f"Random seed set to {RANDOM_STATE} for reproducibility\n")

    # Configuration
    DATA_FILE = 'preprocessed_data.csv'
    OUTPUT_DIR = 'processed_splits_advanced'
    AHU_COL = 'AHU_name'
    LABEL_COL = 'label'

    # Advanced configuration
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    MIN_SAMPLES_FOR_SPLIT = 20
    MIN_SAMPLES_PER_CLASS_PER_AHU = 100  # Augment classes with fewer samples

    # SMOTE options: 'smote', 'borderline', 'adasyn', 'smote_tomek', 'smote_enn'
    SMOTE_METHOD = 'smote'

    # SMOTE ratio options:
    # - 'auto': resample all classes except majority
    # - 'minority': resample only minority class
    # - 'not majority': resample all except majority
    # - float (e.g., 0.5): desired ratio of minority to majority
    # - dict (e.g., {0: 1000, 1: 800}): exact number per class
    SMOTE_RATIO = 'auto'

    K_NEIGHBORS = 5
    APPLY_SMOTE_TO_VAL = False

    print("="*80)
    print("ADVANCED AHU FAULT DETECTION DATASET PREPARATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Test size: {TEST_SIZE}")
    print(f"  Val size: {VAL_SIZE}")
    print(f"  Min samples for split: {MIN_SAMPLES_FOR_SPLIT}")
    print(f"  Min samples per class per AHU: {MIN_SAMPLES_PER_CLASS_PER_AHU}")
    print(f"  SMOTE method: {SMOTE_METHOD}")
    print(f"  SMOTE ratio: {SMOTE_RATIO}")
    print(f"  K neighbors: {K_NEIGHBORS}")

    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print("="*80)
    df = pd.read_csv(DATA_FILE)
    print(f"Total samples: {len(df)}")

    # Convert one-hot encoded AHU columns to single AHU_name column
    print(f"\nConverting one-hot encoded AHU columns to AHU_name...")
    df = convert_onehot_to_ahu_column(df)

    print(f"\nOverall class distribution: {dict(Counter(df[LABEL_COL]))}")

    # Create split creator
    creator = AdvancedAHUSplitCreator(
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        min_samples_for_split=MIN_SAMPLES_FOR_SPLIT,
        min_samples_per_class_per_ahu=MIN_SAMPLES_PER_CLASS_PER_AHU,
        smote_method=SMOTE_METHOD,
        smote_ratio=SMOTE_RATIO,
        k_neighbors=K_NEIGHBORS,
        impute_strategy='constant',
        impute_fill_value=0
    )

    print(f"\nNaN Handling:")
    print(f"  Strategy: constant imputation with value=0")
    print(f"  Reason: NaN values represent seasonal heating/cooling off states")
    print(f"  - Summer: heating columns = NaN → imputed to 0")
    print(f"  - Winter: cooling columns = NaN → imputed to 0")

    # Create splits
    print(f"\n{'='*80}")
    print("CREATING STRATIFIED SPLITS")
    print("="*80)
    splits = creator.create_ahu_splits(df, ahu_col=AHU_COL, label_col=LABEL_COL)

    # Apply SMOTE
    print(f"\n{'='*80}")
    print("APPLYING SMOTE")
    print("="*80)
    splits_with_smote = creator.apply_smote(splits, apply_to_val=APPLY_SMOTE_TO_VAL)

    # Save splits
    print(f"\n{'='*80}")
    print("SAVING SPLITS")
    print("="*80)
    creator.save_splits(splits_with_smote, output_dir=OUTPUT_DIR)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    if 'metadata' in splits_with_smote:
        metadata = splits_with_smote['metadata']

        print(f"\nAHUs processed: {len(metadata['ahu_stats'])}")
        print(f"AHUs excluded: {len(metadata['excluded_ahus'])}")

        if metadata['excluded_ahus']:
            print(f"Excluded AHUs: {', '.join(metadata['excluded_ahus'])}")

        print(f"\nFederated training available for {len(splits_with_smote['federated'])} AHUs")

    print(f"\n{'='*80}")
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()