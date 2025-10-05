import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.metrics import (
    fbeta_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings

class ModelService:
    def __init__(self):
        # Define paths for saving models
        self.MODEL_DIR = "saved_models"
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        
        # Default optimization settings
        self.OPTIMIZATION_SETTINGS = {
            "balanced": {
                "beta": 1.0,
                "threshold": 0.5,
                "scale_pos_weight": 1.0,
                "algorithm": "xgboost"
            },
            "precision": {
                "beta": 0.5,
                "threshold": 0.6,
                "scale_pos_weight": 3.0,
                "algorithm": "lightgbm"
            },
            "recall": {
                "beta": 2.0,
                "threshold": 0.3,
                "scale_pos_weight": 20.0,
                "algorithm": "lightgbm"
            }
        }

    def train_model_user(self, data):
        """
        Trains a machine learning model for planet classification using pre-loaded datasets.
        All model parameters can be customized.
        
        Args:
            data: JSON payload with model parameters
                
        Returns:
            Dictionary with training results and model information
        """
        try:
            # Suppress warnings
            warnings.filterwarnings('ignore')

            # Get request data
            model_type = data.get('model_type', 'TOI')  # Default to TOI
            optimization_type = data.get('optimization_type', 'balanced')  # Default to balanced
            target_column = 'target'  # This is fixed as our target
            model_name = data.get('model_name', f"{model_type}_{optimization_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            print(f"Training {model_type} model with {optimization_type} optimization")
            
            # Load appropriate dataset
            dataset_path = f"src/utils/datasets/{model_type}.csv"
            if not os.path.exists(dataset_path):
                print(f"Dataset not found at: {dataset_path}")
                # Try alternative path
                dataset_path = f"utils/datasets/{model_type}.csv"
                if not os.path.exists(dataset_path):
                    return {
                        "status": "error",
                        "message": f"Dataset file not found for {model_type}. Looked in: 'src/utils/datasets/' and 'utils/datasets/'"
                    }, 404
                    
            print(f"Loading dataset from: {dataset_path}")
            df = pd.read_csv(dataset_path)
            print(f"Dataset loaded with shape: {df.shape}")
            
            # Create target variable based on model type
            if model_type == 'KOI':
                # For KOI: CONFIRMED or CANDIDATE = 1, others = 0
                if 'koi_disposition' in df.columns:
                    df[target_column] = df['koi_disposition'].apply(
                        lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
                    )
                else:
                    return {
                        "status": "error",
                        "message": "Column 'koi_disposition' not found in KOI dataset"
                    }, 400
            else:  # TOI
                # For TOI: CP or KP = 1, FP = 0, others = NaN
                if 'tfopwg_disp' in df.columns:
                    df[target_column] = df['tfopwg_disp'].apply(
                        lambda x: 1 if x in ['CP', 'KP'] else 0 if x == 'FP' else np.nan
                    )
                    # Remove rows with NaN target
                    df = df.dropna(subset=[target_column])
                    # Convert target to int
                    df[target_column] = df[target_column].astype(int)
                else:
                    return {
                        "status": "error",
                        "message": "Column 'tfopwg_disp' not found in TOI dataset"
                    }, 400
            
            # Define required features based on model type
            if model_type == 'TOI':
                required_features = [
                    "pl_trandurherr1", "eng_transit_probability", "pl_trandurherr2", "eng_prad_srad_ratio",
                    "pl_orbpererr1", "pl_orbper", "eng_period_duration_ratio", "eng_duration_period_ratio",
                    "pl_tranmiderr1", "pl_trandeperr2", "pl_tranmid", "pl_trandep", "pl_trandurh",
                    "pl_trandeperr1", "st_tmagerr2", "st_disterr2", "st_loggerr2", "st_disterr1",
                    "st_dist", "st_teff", "st_tmagerr1", "st_rad", "st_tefferr2"
                ]
            else:  # KOI
                required_features = [
                    'koi_prad', 'koi_dor', 'koi_ror', 'koi_num_transits', 'koi_duration_err1', 'koi_prad_err1',
                    'koi_period_err2', 'koi_srad_err1', 'koi_insol', 'eng_transit_probability', 'koi_model_snr',
                    'koi_srho', 'koi_max_mult_ev', 'koi_teq'
                ]
                
            # Check if all required features are in the dataset
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                return {
                    "status": "error",
                    "message": f"Missing required features in dataset: {missing_features}"
                }, 400
                
            # Filter dataset to only include required features and target
            features_to_use = required_features.copy()
            
            # Prepare data
            X = df[features_to_use]
            y = df[target_column]
            
            # Check class distribution
            class_counts = y.value_counts()
            print(f"Class distribution: {class_counts.to_dict()}")
            
            if len(class_counts) < 2:
                return {
                    "status": "error",
                    "message": f"Only one class found in target column. Need at least 2 classes."
                }, 400
                
            # Get all data preparation parameters with defaults
            do_oversample = data.get('do_oversample', True)
            min_samples_for_oversample = data.get('min_samples_for_oversample', 5)
            test_size = data.get('test_size', 0.3)
            validation_size = data.get('validation_size', 0.5)
            random_state = data.get('random_state', 42)
            
            # Handle class imbalance by oversampling the minority class
            if do_oversample and len(class_counts) == 2:
                majority_class = class_counts.idxmax()
                minority_class = class_counts.idxmin()
                
                if class_counts[minority_class] < min_samples_for_oversample:
                    # Oversample minority class
                    majority_data = df[df[target_column] == majority_class]
                    minority_data = df[df[target_column] == minority_class]
                    
                    # Upsample minority class
                    minority_upsampled = resample(
                        minority_data,
                        replace=True,     # sample with replacement
                        n_samples=len(majority_data),  # match majority class
                        random_state=random_state
                    )
                    
                    # Combine majority with upsampled minority
                    df = pd.concat([majority_data, minority_upsampled])
                    
                    # Update X and y
                    X = df[features_to_use]
                    y = df[target_column]
                    
                    print(f"Performed oversampling. New class distribution: {y.value_counts().to_dict()}")
                    
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_test, y_test,
                test_size=validation_size,
                random_state=random_state,
                stratify=y_test
            )
            
            print(f"Data split into train ({len(X_train)}), validation ({len(X_val)}), test ({len(X_test)}) sets")

            # Get base settings from optimization type
            settings = self.OPTIMIZATION_SETTINGS[optimization_type].copy()

            # Allow overriding of core settings
            beta = data.get('beta', settings['beta'])
            threshold = data.get('threshold', settings['threshold'])
            settings['beta'] = beta
            settings['threshold'] = threshold

            # Get algorithm - allow overriding but default based on optimization type
            algorithm = data.get('algorithm', settings['algorithm'])
            settings['algorithm'] = algorithm

            # Error penalty weights
            false_positive_weight = data.get('false_positive_weight', 1.8 if optimization_type == 'precision' else 1.0)
            false_negative_weight = data.get('false_negative_weight', 450 if optimization_type == 'recall' else 1.0)
            weight_value = data.get('weight_value', 
                                false_positive_weight if optimization_type == 'precision' 
                                else false_negative_weight if optimization_type == 'recall'
                                else 1.0)

            # Train model based on algorithm selection
            if algorithm == 'xgboost':
                # XGBoost parameters - expose ALL parameters
                params = {
                    'booster': data.get('booster', 'gbtree'),
                    'objective': data.get('objective', 'binary:logistic'),
                    'max_depth': data.get('max_depth', 4),
                    'learning_rate': data.get('learning_rate', 0.1),
                    'min_child_weight': data.get('min_child_weight', 2),
                    'gamma': data.get('gamma', 0),
                    'subsample': data.get('subsample', 0.8),
                    'colsample_bytree': data.get('colsample_bytree', 0.8),
                    'colsample_bylevel': data.get('colsample_bylevel', 1.0),
                    'colsample_bynode': data.get('colsample_bynode', 1.0),
                    'reg_lambda': data.get('reg_lambda', 1.0),
                    'reg_alpha': data.get('reg_alpha', 0.01),
                    'n_estimators': data.get('n_estimators', 200),
                    'scale_pos_weight': data.get('scale_pos_weight', settings['scale_pos_weight']),
                    'random_state': data.get('random_state', random_state),
                    'tree_method': data.get('tree_method', 'auto'),
                    'importance_type': data.get('importance_type', 'gain'),
                    'max_delta_step': data.get('max_delta_step', 0),
                    'grow_policy': data.get('grow_policy', 'depthwise')
                }
                
                # Create early stopping callback if requested
                early_stopping_rounds = data.get('early_stopping_rounds', None)
                callbacks = []
                if early_stopping_rounds:
                    callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
                
                print(f"Training XGBoost model with parameters: {params}")
                model = xgb.XGBClassifier(**params)
                
                eval_set = [(X_train, y_train), (X_val, y_val)] if early_stopping_rounds else None
                if eval_set:
                    model.fit(X_train, y_train, eval_set=eval_set)
                else:
                    model.fit(X_train, y_train)
                
            elif algorithm == 'lightgbm':
                # Create custom asymmetric objective function if requested
                use_custom_objective = data.get('use_custom_objective', True)
                
                if use_custom_objective:
                    # Custom asymmetric objective function
                    def custom_asymmetric_objective(preds, train_data):
                        y_true = train_data.get_label()
                        y_pred = 1.0 / (1.0 + np.exp(-preds))
                        grad = y_pred - y_true
                        hess = y_pred * (1.0 - y_pred)
        
                        if optimization_type == "precision":
                            grad = np.where(y_true < 0.5, weight_value * grad, grad)
                            hess = np.where(y_true < 0.5, weight_value * hess, hess)
                        else:  # recall or default
                            grad = np.where(y_true > 0.5, weight_value * grad, grad)
                            hess = np.where(y_true > 0.5, weight_value * hess, hess)
        
                        return grad, hess
                    
                    objective = custom_asymmetric_objective
                else:
                    objective = data.get('objective', 'binary')

                # LightGBM parameters - expose ALL parameters
                params = {
                    'boosting_type': data.get('boosting_type', 'gbdt'),
                    'objective': objective,
                    'metric': data.get('metric', 'auc'),
                    'learning_rate': data.get('learning_rate', 0.1),
                    'num_leaves': data.get('num_leaves', 31),
                    'max_depth': data.get('max_depth', 6),
                    'min_data_in_leaf': data.get('min_data_in_leaf', 1),
                    'min_child_samples': data.get('min_child_samples', 1),
                    'min_child_weight': data.get('min_child_weight', 1e-3),
                    'feature_fraction': data.get('feature_fraction', 0.9),
                    'bagging_fraction': data.get('bagging_fraction', 0.8),
                    'bagging_freq': data.get('bagging_freq', 5),
                    'lambda_l1': data.get('lambda_l1', 0.1),
                    'lambda_l2': data.get('lambda_l2', 0.2),
                    'scale_pos_weight': data.get('scale_pos_weight', settings['scale_pos_weight']),
                    'subsample_for_bin': data.get('subsample_for_bin', 200000),
                    'subsample': data.get('subsample', 1.0),
                    'min_split_gain': data.get('min_split_gain', 0.0),
                    'seed': data.get('seed', random_state),
                    'verbose': data.get('verbose', -1),
                    'deterministic': data.get('deterministic', False),
                    'early_stopping_round': data.get('early_stopping_round', None)
                }
                
                # Remove objective from params if it's a custom function
                if use_custom_objective:
                    if 'objective' in params and callable(params['objective']):
                        del params['objective']
        
                print(f"Training LightGBM model with parameters: {params}")
                # Create LightGBM datasets
                train_data_lgb = lgb.Dataset(X_train, label=y_train)
                valid_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
        
                # Set up training parameters
                num_boost_round = data.get('num_boost_round', 1000)
                early_stopping_rounds = data.get('early_stopping_rounds', 50)
                verbose_eval = data.get('verbose_eval', False)
                
                # Train LightGBM model with early stopping
                model = lgb.train(
                    params,
                    train_data_lgb,
                    num_boost_round=num_boost_round,
                    valid_sets=[valid_data_lgb],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose_eval)
                    ] if early_stopping_rounds else None
                )
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported algorithm: {algorithm}. Use 'xgboost' or 'lightgbm'."
                }, 400

            # Create threshold for classification
            threshold = data.get('threshold', settings['threshold'])

            # Option to find optimal threshold
            find_optimal_threshold = data.get('find_optimal_threshold', False)
            if find_optimal_threshold:
                print("Finding optimal classification threshold...")
                # Test a range of thresholds
                thresholds = np.linspace(0.1, 0.9, 17)  # 0.1 to 0.9 in 0.05 increments
                threshold_results = []
                
                for thresh in thresholds:
                    if algorithm == 'xgboost':
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                    else:
                        y_pred_proba = model.predict(X_val)
                        
                    y_pred = (y_pred_proba >= thresh).astype(int)
                    
                    if sum(y_pred) > 0:  # Avoid division by zero
                        precision = precision_score(y_val, y_pred, zero_division=0)
                        recall = recall_score(y_val, y_pred, zero_division=0)
                        f_beta_score_val = fbeta_score(y_val, y_pred, beta=beta, zero_division=0)
                        threshold_results.append((thresh, precision, recall, f_beta_score_val))
                
                if threshold_results:
                    # Find threshold with best F-beta score
                    threshold_df = pd.DataFrame(
                        threshold_results,
                        columns=['Threshold', 'Precision', 'Recall', f'F{beta} Score']
                    )
                    
                    best_idx = threshold_df[f'F{beta} Score'].idxmax()
                    threshold = threshold_df.iloc[best_idx]['Threshold']
                    print(f"Optimal threshold found: {threshold:.2f}")
                else:
                    print("Could not find optimal threshold, using default.")

            # Evaluate model
            def evaluate_dataset(X, y, dataset_name):
                try:
                    if algorithm == 'xgboost':
                        y_pred_proba = model.predict_proba(X)[:, 1]
                        y_pred = (y_pred_proba >= threshold).astype(int)
                    else:
                        y_pred_proba = model.predict(X)
                        y_pred = (y_pred_proba >= threshold).astype(int)

                    # Calculate metrics safely
                    accuracy = accuracy_score(y, y_pred)
                    precision = precision_score(y, y_pred, zero_division=0)
                    recall = recall_score(y, y_pred, zero_division=0)
                    f1 = f1_score(y, y_pred, zero_division=0)
                    f_beta = fbeta_score(y, y_pred, beta=beta, zero_division=0)
                    
                    # Calculate ROC AUC safely
                    try:
                        roc_auc = roc_auc_score(y, y_pred_proba)
                    except:
                        roc_auc = 0.5
                    
                    # Force 2x2 confusion matrix with explicit labels
                    cm = confusion_matrix(y, y_pred, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()

                    return {
                        'dataset': dataset_name,
                        'metrics': {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1': float(f1),
                            f'f{beta}': float(f_beta),
                            'roc_auc': float(roc_auc)
                        },
                        'confusion_matrix': {
                            'true_negative': int(tn),
                            'false_positive': int(fp),
                            'false_negative': int(fn),
                            'true_positive': int(tp)
                        }
                    }
                except Exception as e:
                    print(f"Error evaluating {dataset_name}: {str(e)}")
                    return {
                        'dataset': dataset_name,
                        'metrics': {
                            'accuracy': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1': 0.0,
                            f'f{beta}': 0.0,
                            'roc_auc': 0.5
                        },
                        'confusion_matrix': {
                            'true_negative': 0,
                            'false_positive': 0,
                            'false_negative': 0,
                            'true_positive': 0
                        }
                    }

            # Evaluate on all datasets
            print("Evaluating model performance...")
            train_eval = evaluate_dataset(X_train, y_train, "training")
            val_eval = evaluate_dataset(X_val, y_val, "validation")
            test_eval = evaluate_dataset(X_test, y_test, "test")

            # Feature importance
            try:
                if algorithm == 'xgboost':
                    importance = model.feature_importances_
                else:
                    importance = model.feature_importance(importance_type='gain')

                feature_importance = {
                    feature: float(imp) for feature, imp in zip(features_to_use, importance)
                }
                feature_importance = dict(sorted(
                    feature_importance.items(),
                    key=lambda item: item[1],
                    reverse=True
                ))
            except Exception as e:
                print(f"Error calculating feature importance: {e}")
                feature_importance = {feature: 1.0 for feature in features_to_use}

            # Get visualization options
            create_visualizations = data.get('create_visualizations', True)
            visualization_path = data.get('visualization_path', os.path.join("src", "visualizations"))
            
            # Create visualizations if requested
            if create_visualizations:
                os.makedirs(visualization_path, exist_ok=True)
                
                # Metrics comparison
                metrics_df = pd.DataFrame({
                    'Training': [
                        train_eval['metrics']['accuracy'],
                        train_eval['metrics']['precision'],
                        train_eval['metrics']['recall'],
                        train_eval['metrics']['f1'],
                        train_eval['metrics'][f'f{beta}'],
                        train_eval['metrics']['roc_auc']
                    ],
                    'Validation': [
                        val_eval['metrics']['accuracy'],
                        val_eval['metrics']['precision'],
                        val_eval['metrics']['recall'],
                        val_eval['metrics']['f1'],
                        val_eval['metrics'][f'f{beta}'],
                        val_eval['metrics']['roc_auc']
                    ],
                    'Test': [
                        test_eval['metrics']['accuracy'],
                        test_eval['metrics']['precision'],
                        test_eval['metrics']['recall'],
                        test_eval['metrics']['f1'],
                        test_eval['metrics'][f'f{beta}'],
                        test_eval['metrics']['roc_auc']
                    ]
                }, index=['Accuracy', 'Precision', 'Recall', 'F1 Score', f'F{beta} Score', 'ROC AUC'])
                
                try:
                    plt.figure(figsize=(10, 6))
                    metrics_df.plot(kind='bar')
                    plt.title(f'{model_type} Model Performance (Optimization: {optimization_type})')
                    plt.ylabel('Score')
                    plt.ylim([0, 1.05])
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(visualization_path, f'{model_name}_metrics.png'))
                    plt.close()
                    
                    # Feature importance plot
                    if len(feature_importance) > 0:
                        plt.figure(figsize=(12, 8))
                        importance_df = pd.DataFrame({
                            'Feature': list(feature_importance.keys()),
                            'Importance': list(feature_importance.values())
                        }).head(20)  # Top 20 features
                        
                        plt.barh(importance_df['Feature'], importance_df['Importance'])
                        plt.title('Top 20 Feature Importance')
                        plt.xlabel('Importance Score')
                        plt.tight_layout()
                        plt.savefig(os.path.join(visualization_path, f'{model_name}_importance.png'))
                        plt.close()
                except Exception as vis_error:
                    print(f"Error creating visualizations: {vis_error}")

            # Prepare model info and evaluation data for saving
            model_info = {
                'model_name': model_name,
                'model_type': model_type,
                'optimization_type': optimization_type,
                'algorithm': algorithm,
                'beta': float(beta),
                'threshold': float(threshold),
                'weight_value': float(weight_value),
                'false_positive_weight': float(false_positive_weight),
                'false_negative_weight': float(false_negative_weight),
                'features': features_to_use,
                'feature_count': len(features_to_use),
                'class_distribution': {str(k): int(v) for k, v in y.value_counts().items()},
                'training_params': params
            }
            
            evaluation_data = {
                'training': train_eval,
                'validation': val_eval,
                'test': test_eval
            }
            
            # Create response data
            response_data = {
                'status': 'success',
                'message': f'Model trained and saved as {model_name}',
                'model_info': model_info,
                'evaluation': evaluation_data,
                'feature_importance': feature_importance
            }
            
            # Create standard directory structure
            model_save_dir = os.path.join("src", "trained_models", model_type, f"{optimization_type}_model")
            os.makedirs(model_save_dir, exist_ok=True)
            print(f"Created model directory: {model_save_dir}")
            
            # Save the model and metrics in both locations
            # 1. Save to saved_models with custom name (for user access)
            custom_model_path = os.path.join(self.MODEL_DIR, f"{model_name}.pkl")
            print(f"Saving model to custom path: {custom_model_path}")
            
            model_data_to_save = {
                'model': model,
                'model_type': model_type,
                'optimization_type': optimization_type,
                'features': features_to_use,
                'settings': settings,
                'threshold': threshold,
                'algorithm': algorithm,
                'model_info': model_info,
                'evaluation': evaluation_data,
                'feature_importance': feature_importance,
                'params': params
            }
            
            with open(custom_model_path, 'wb') as f:
                pickle.dump(model_data_to_save, f)
                
            # 2. Save to standard location (for prediction endpoint)
            standard_model_path = os.path.join(model_save_dir, "model.pkl")
            print(f"Saving model to standard path: {standard_model_path}")
            with open(standard_model_path, 'wb') as f:
                pickle.dump(model_data_to_save, f)
                
            # 3. Also save metrics for quick access
            metrics_path = os.path.join(model_save_dir, "metrics.json")
            print(f"Saving metrics to: {metrics_path}")
            with open(metrics_path, 'w') as f:
                json.dump({
                    'model_info': model_info,
                    'evaluation': evaluation_data,
                    'feature_importance': feature_importance
                }, f, indent=4)
            
            print(f"Model and metrics saved successfully")
            
            # Return results
            return response_data

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Error in train_model_user: {e}")
            print(error_traceback)
            return {
                'status': 'error',
                'message': str(e),
                'traceback': error_traceback
            }, 500


    def predict(self, data):
        """
        Makes predictions using a trained model.
        
        Args:
            data: JSON payload with prediction data and model parameters
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get request data
            model_type = data.get('model_type', 'TOI')  # TOI or KOI
            optimization_type = data.get('optimization_type', 'balanced')  # balanced, precision, or recall
            model_name = data.get('model_name')
            
            # Get input data for prediction
            input_data = data.get('data', [])
            if not input_data:
                return {
                    'status': 'error',
                    'message': 'No data provided for prediction'
                }, 400
                
            # Default paths for model files based on model type and optimization
            model_dir = f"src/utils/trained_models/{model_type}/{optimization_type}_model"
            print(f"Looking for model in: {model_dir}")
            
            # Find the model path - prefer the path from model_name if provided
            model_path = None
            
            if model_name:
                # First, check if a specific model name was provided and exists in saved_models
                custom_model_path = os.path.join(self.MODEL_DIR, f"{model_name}.pkl")
                print(f"Checking for custom model at: {custom_model_path}")
                if os.path.exists(custom_model_path):
                    model_path = custom_model_path
                    print(f"Found custom model: {model_path}")
                else:
                    print(f"Custom model not found at: {custom_model_path}")
            
            # Otherwise, use the pre-trained models based on type and optimization
            if not model_path:
                # Try to find model file - check both .pkl and .json formats
                pkl_path = os.path.join(model_dir, "model.pkl")
                json_path = os.path.join(model_dir, "model.json")
                
                print(f"Checking for pkl model at: {pkl_path}")
                print(f"Checking for json model at: {json_path}")
                
                if os.path.exists(pkl_path):
                    model_path = pkl_path
                    print(f"Found pkl model: {model_path}")
                elif os.path.exists(json_path):
                    model_path = json_path
                    print(f"Found json model: {model_path}")
                else:
                    # List all files in the directory for debugging
                    if os.path.exists(model_dir):
                        print(f"Directory exists, contents: {os.listdir(model_dir)}")
                    else:
                        print(f"Directory does not exist: {model_dir}")
                        # Try to list parent directory
                        parent_dir = os.path.dirname(model_dir)
                        if os.path.exists(parent_dir):
                            print(f"Parent directory exists, contents: {os.listdir(parent_dir)}")
                        
                    return {
                        'status': 'error',
                        'message': f'No model found for {model_type} with {optimization_type} optimization. Please check directory structure and model files.'
                    }, 404
            
            # Set the default threshold based on optimization type
            if optimization_type == 'precision':
                threshold = 0.6
            elif optimization_type == 'recall':
                threshold = 0.3
            else:  # balanced
                threshold = 0.5
                
            # Load the model based on file extension
            is_lightgbm = False
            try:
                if model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as f:
                        print(f"Loading pickle file from {model_path}")
                        model_data = pickle.load(f)
                        print(f"Model type: {type(model_data)}")
                        
                        # Check if it's a dictionary containing a model or a direct model object
                        if isinstance(model_data, dict) and 'model' in model_data:
                            # It's a dictionary with model metadata
                            model = model_data['model']
                            print(f"Model found in dictionary, model type: {type(model)}")
                            threshold = model_data.get('threshold', threshold)
                            is_lightgbm = model_data.get('algorithm', '').lower() == 'lightgbm'
                            print(f"Using threshold: {threshold}, is_lightgbm: {is_lightgbm}")
                        else:
                            # It's the model object directly
                            model = model_data
                            print(f"Model is direct object, model type: {type(model)}")
                            # Determine if it's LightGBM based on the model class
                            is_lightgbm = 'lightgbm' in str(type(model)).lower()
                            print(f"Using default threshold: {threshold}, is_lightgbm: {is_lightgbm}")
                            
                elif model_path.endswith('.json'):
                    # For LightGBM models saved as JSON
                    print(f"Loading LightGBM model from JSON: {model_path}")
                    model = lgb.Booster(model_file=model_path)
                    is_lightgbm = True
                    print(f"LightGBM model loaded, using default threshold: {threshold}")
                else:
                    return {
                        'status': 'error',
                        'message': f'Unknown model format: {model_path}'
                    }, 400
            except Exception as load_error:
                print(f"Error loading model: {load_error}")
                import traceback
                print(traceback.format_exc())
                return {
                    'status': 'error',
                    'message': f'Error loading model from {model_path}: {str(load_error)}'
                }, 500
                
            # Identify which feature set to use based on model type
            if model_type == 'TOI':
                # TOI features (23 total - reduced set without leaky features)
                required_features = [
                        "pl_trandurherr1", "eng_transit_probability","pl_trandurherr2", "eng_prad_srad_ratio",
                        "pl_orbpererr1", "pl_orbper", "eng_period_duration_ratio", "eng_duration_period_ratio",
                        "pl_tranmiderr1", "pl_trandeperr2", "pl_tranmid", "pl_trandep", "pl_trandurh",
                        "pl_trandeperr1", "st_tmagerr2", "st_disterr2", "st_loggerr2", "st_disterr1",
                        "st_dist", "st_teff", "st_tmagerr1", "st_rad", "st_tefferr2"
                    ]
            else:  # KOI features (14 total - reduced set without leaky features)
                required_features = [
                        'koi_prad', 'koi_dor', 'koi_ror', 'koi_num_transits', 'koi_duration_err1', 'koi_prad_err1',
                        'koi_period_err2', 'koi_srad_err1', 'koi_insol', 'eng_transit_probability', 'koi_model_snr',
                        'koi_srho', 'koi_max_mult_ev', 'koi_teq'
                    ]

            # Convert input data to DataFrame
            input_df = pd.DataFrame(input_data)
            print(f"Input data columns: {input_df.columns.tolist()}")
            
            # Check if all required features are present
            missing_features = [f for f in required_features if f not in input_df.columns]
            if missing_features:
                return {
                    'status': 'error', 
                    'message': f'Missing features: {missing_features}. Required features for {model_type} models are: {required_features}'
                }, 400
                
            # Extract only the required features
            X = input_df[required_features]
            print(f"Using {len(required_features)} features for prediction")
            
            # Make prediction based on algorithm
            try:
                if is_lightgbm:
                    print("Using LightGBM prediction")
                    probabilities = model.predict(X)
                else:  # xgboost or other sklearn-compatible
                    # Check if model has predict_proba method
                    if hasattr(model, 'predict_proba'):
                        print("Using model.predict_proba")
                        probabilities = model.predict_proba(X)[:, 1]
                    else:
                        # Fallback to predict for models without predict_proba
                        print("Using model.predict (no predict_proba available)")
                        probabilities = model.predict(X)
                        
                print(f"Prediction shape: {probabilities.shape if hasattr(probabilities, 'shape') else 'scalar'}")
                print(f"First few probabilities: {probabilities[:3] if hasattr(probabilities, '__getitem__') else probabilities}")
                
                predictions = (probabilities >= threshold).astype(int)
                print(f"Using threshold {threshold}, made {len(predictions)} predictions")
                
            except Exception as pred_error:
                print(f"Error during prediction: {pred_error}")
                import traceback
                print(traceback.format_exc())
                return {
                    'status': 'error',
                    'message': f'Error making predictions: {str(pred_error)}'
                }, 500
                
            # Return results
            results = []
            for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
                results.append({
                    'id': i,
                    'probability': float(prob),
                    'prediction': int(pred),
                    'class': 'Planet Candidate/Confirmed' if pred == 1 else 'False Positive',
                    'threshold_used': float(threshold)
                })
                
            return {
                'status': 'success',
                'model_type': model_type,
                'optimization_type': optimization_type,
                'threshold': float(threshold),
                'prediction_count': len(results),
                'results': results
            }
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Unexpected error in predict method: {e}")
            print(error_traceback)
            return {
                'status': 'error',
                'message': str(e),
                'traceback': error_traceback
            }, 500