import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage
from datetime import datetime
from scipy.spatial import KDTree
from src.config.configloading import load_config

# Legend for the comments in this file:
#   [ORIGINAL] = code kept as in eval_orig.py (the reason it was kept is given)
#   [EDIT]     = changed or added compared with eval_orig.py (the reason is given)


def compute_overlap_metric(
    label: np.ndarray,
    output: np.ndarray,
    d: float = 0,
    voxel_spacing: tuple = (0.37695312, 0.37695312, 0.5),
    threshold_label: float = 0.5,
    threshold_output: float = 0.5
):
    """
    Computes the Overlap Metric (Ot(d)) between a binary label and a predicted output.
    
    Args:
        label (numpy.ndarray): Ground truth binary array (2D or 3D).
        output (numpy.ndarray): Predicted binary array (same shape as label).
        d (float): Distance threshold in mm (0 for Dice, 1 or 2 for relaxed overlap).
        voxel_spacing (tuple): Physical voxel spacing in mm (x,y,z). Default matches CCTA data.
        threshold_label (float): Binarization threshold for label (default: 0.5).
        threshold_output (float): Binarization threshold for output (default: 0.5).
    
    Returns:
        float: Overlap score Ot(d) ∈ [0, 1].
    """
    # Binarize inputs
    label_bin = (label >= threshold_label).astype(np.uint8)
    output_bin = (output >= threshold_output).astype(np.uint8)
    
    # If d=0, compute Dice score (Ot(0))
    if d == 0:
        intersection = np.sum(label_bin & output_bin)
        union = np.sum(label_bin) + np.sum(output_bin)
        return (2 * intersection) / union if union != 0 else 0.0
    
    # Convert mm distance to voxel units using the smallest spacing dimension
    d_voxels = d / min(voxel_spacing)
    
    # Get coordinates of foreground points in label and output
    label_points = np.argwhere(label_bin > 0)
    output_points = np.argwhere(output_bin > 0)
    
    # If either set is empty, return 0 (no overlap)
    if len(label_points) == 0 or len(output_points) == 0:
        return 0.0
    
    # Build KDTree for the output points (prediction)
    kdtree = KDTree(output_points)
    
    # Find TPR(d): Label points within distance d_voxels of any output point
    dist_label_to_output, _ = kdtree.query(label_points, distance_upper_bound=d_voxels)
    tpr_d = np.sum(dist_label_to_output <= d_voxels)
    fn_d = len(label_points) - tpr_d
    
    # Build KDTree for the label points (ground truth)
    kdtree_label = KDTree(label_points)
    
    # Find TPM(d): Output points within distance d_voxels of any label point
    dist_output_to_label, _ = kdtree_label.query(output_points, distance_upper_bound=d_voxels)
    tpm_d = np.sum(dist_output_to_label <= d_voxels)
    fp_d = len(output_points) - tpm_d
    
    # Compute Ot(d)
    ot_d = (tpm_d + tpr_d) / (tpm_d + tpr_d + fn_d + fp_d)
    return ot_d


def compute_chamfer_distance(
    label: np.ndarray,
    output: np.ndarray,
    voxel_spacing: tuple = (0.37695312, 0.37695312, 0.5),
    threshold_label: float = 0.5,
    threshold_output: float = 0.5,
    physical_units: bool = True
) -> float:
    """
    Computes the symmetric Chamfer Distance (L2) between two binary volumes.
    
    Args:
        label (np.ndarray): Ground truth binary array (2D/3D).
        output (np.ndarray): Predicted binary array (same shape as label).
        voxel_spacing (tuple): Physical voxel spacing in mm (x,y,z). Default matches CCTA data.
        threshold_label (float): Binarization threshold for label. Default: 0.5.
        threshold_output (float): Binarization threshold for output. Default: 0.5.
        physical_units (bool): If True, returns distance in mm. If False, in voxels.
    
    Returns:
        float: Chamfer Distance (L2) in mm or voxels.
    """

    # Binarize inputs
    label_bin = (label >= threshold_label).astype(np.uint8)
    output_bin = (output >= threshold_output).astype(np.uint8)
    
    # Get coordinates of foreground points
    label_points = np.argwhere(label_bin > 0)
    output_points = np.argwhere(output_bin > 0)
    
    # Handle empty cases
    if len(label_points) == 0 or len(output_points) == 0:
        return np.inf  # No overlap
    
    # Scale coordinates to physical units if requested
    if physical_units:
        label_points = label_points * np.array(voxel_spacing)
        output_points = output_points * np.array(voxel_spacing)
    
    # Build KDTrees
    kdtree_label = KDTree(label_points)
    kdtree_output = KDTree(output_points)
    
    # Label → Output distances
    dist_label_to_output, _ = kdtree_output.query(label_points)
    term1 = np.mean(dist_label_to_output ** 2)
    
    # Output → Label distances
    dist_output_to_label, _ = kdtree_label.query(output_points)
    term2 = np.mean(dist_output_to_label ** 2)
    
    # Symmetric Chamfer Distance (ℓ2)
    chamfer_distance = (term1 + term2) ** 0.5
    return chamfer_distance


def compute_iou(
    label: np.ndarray,
    output: np.ndarray,
    threshold_label: float = 0.5,
    threshold_output: float = 0.5
) -> float:
    """Computes Intersection over Union (IoU) between binary volumes."""
    label_bin = (label >= threshold_label).astype(np.uint8)
    output_bin = (output >= threshold_output).astype(np.uint8)
    
    intersection = np.sum(label_bin & output_bin)
    union = np.sum(label_bin | output_bin)
    
    return intersection / union if union != 0 else 0.0


# [EDIT] New helper (not in eval_orig.py).
# Reason: the NeCA paper (Sec. 2.5), whose protocol SDF-CAR follows, removes disconnected
# fragments smaller than 25 voxels from the reconstruction before computing any metric.
def remove_small_components(
    output: np.ndarray,
    threshold: float = 0.5,
    min_voxels: int = 25
) -> np.ndarray:
    """Zeroes predicted foreground components smaller than min_voxels (full connectivity)."""
    mask = output >= threshold
    structure = ndimage.generate_binary_structure(output.ndim, output.ndim)
    labeled, n_components = ndimage.label(mask, structure=structure)
    if n_components == 0:
        return output
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_components + 1))
    small_ids = np.where(sizes < min_voxels)[0] + 1
    cleaned = output.copy()
    cleaned[np.isin(labeled, small_ids)] = 0
    return cleaned


def compute_cldice(
    label: np.ndarray,
    output: np.ndarray,
    threshold_label: float = 0.5,
    threshold_output: float = 0.5
) -> float:
    """
    Computes centerline Dice (clDice) for tubular structures.
    
    Uses the standard formula:
        Tprec = |Skel(V_P) ∩ V_L| / |Skel(V_P)|   (topology precision)
        Tsens = |Skel(V_L) ∩ V_P| / |Skel(V_L)|   (topology sensitivity)
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    """
    try:
        from skimage.morphology import skeletonize
        
        label_bin = (label >= threshold_label).astype(np.uint8)
        output_bin = (output >= threshold_output).astype(np.uint8)
        
        if np.sum(label_bin) == 0 or np.sum(output_bin) == 0:
            return 0.0
        
        label_skel = skeletonize(label_bin).astype(np.uint8)
        output_skel = skeletonize(output_bin).astype(np.uint8)
        
        if np.sum(output_skel) == 0 or np.sum(label_skel) == 0:
            return 0.0
        
        # [EDIT] Standard clDice (Shit et al., 2021) replaces eval_orig.py's skeleton-vs-skeleton Dice
        #        (2*|SkelL & SkelP| / (|SkelL| + |SkelP|)).
        # Reason: the original needed the two 1-voxel-wide skeletons to overlap voxel for voxel, so a
        #         one-voxel shift scored ~0. The standard version checks each skeleton against the other
        #         volume. It is the metric the paper cites, but it is more forgiving, so old and new
        #         clDice numbers are not comparable.
        # Topology Precision: fraction of prediction's skeleton inside GT volume
        tprec = np.sum(output_skel & label_bin) / np.sum(output_skel)
        
        # Topology Sensitivity: fraction of GT's skeleton inside prediction volume
        tsens = np.sum(label_skel & output_bin) / np.sum(label_skel)
        
        # clDice = harmonic mean of Tprec and Tsens
        if tprec + tsens == 0:
            return 0.0
        return 2.0 * tprec * tsens / (tprec + tsens)
    except:  # [ORIGINAL] kept. Reason: not part of this investigation. Caution: it hides real errors as a 0.0 score.
        return 0.0


# [ORIGINAL] Whole-volume mean L1 error, exactly as in eval_orig.py.
# Reason: restored to the original definition. Caution: most of the volume is empty background, so the
#         values are tiny (~1e-3) and will not match the paper's ~0.1 scale. If you need that scale,
#         use the region-only alternative commented out below and state it in your write-up.
def compute_reconstruction_error(label: np.ndarray, output: np.ndarray) -> float:
    """Computes reconstruction error (L1 norm) between volumes."""
    return np.mean(np.abs(label - output))


# [EDIT, disabled alternative] Region-only reError (union of foreground voxels).
# Reason: avoids the background dominating the mean, which gives values near the paper's scale.
# def compute_reconstruction_error(
#     label: np.ndarray,
#     output: np.ndarray,
#     threshold: float = 0.5
# ) -> float:
#     label_bin = (label >= threshold).astype(np.uint8)
#     output_bin = (output >= threshold).astype(np.uint8)
#     roi_mask = (label_bin | output_bin).astype(bool)  # union of foreground regions
#     if np.sum(roi_mask) == 0:
#         return 0.0
#     return np.mean(np.abs(label[roi_mask] - output[roi_mask]))


# [EDIT] Plain MSE (no sqrt). eval_orig.py returned np.sqrt(np.mean(...)), i.e. RMSE.
# Reason: the paper reports reMSE in units of 1e-4, which is a plain-MSE scale (NeCA reports ~2.7e-4 for
#         RCA). RMSE values are not comparable. Still computed over the whole volume, as in the original.
def compute_remse(label: np.ndarray, output: np.ndarray) -> float:
    """Computes voxel-wise mean squared error (reMSE), not root."""
    return np.mean((label - output) ** 2)


# [EDIT, disabled alternative] Region-only reMSE (union of foreground voxels).
# Reason: same idea as the region-only reError above. Not used by default.
# def compute_remse(label, output, threshold=0.5):
#     roi_mask = ((label >= threshold) | (output >= threshold))
#     if np.sum(roi_mask) == 0:
#         return 0.0
#     return np.mean((label[roi_mask] - output[roi_mask]) ** 2)


def compute_all_metrics(
    label: np.ndarray,
    output: np.ndarray,
    voxel_spacing: tuple = (0.8, 0.8, 0.8),  # [ORIGINAL] kept. Reason: not changed in this investigation (scales Chamfer distance and volumes).
    threshold_label: float = 0.5,   # [EDIT] was 0.8 in eval_orig.py. Reason: NeCA binarises the occupancy at 0.5 (final output is a sigmoid).
    threshold_output: float = 0.5,  # [EDIT] was 0.8 in eval_orig.py. Reason: same as above. This raises Dice, IoU, clDice and changes CD.
    apply_rotation: bool = True,
    min_component_voxels: int = 25  # [EDIT] new argument. Reason: NeCA's cleanup of fragments < 25 voxels (0 turns it off).
) -> dict:
    """
    Compute all evaluation metrics for a pair of volumes.
    
    Args:
        label (np.ndarray): Ground truth volume
        output (np.ndarray): Predicted volume
        voxel_spacing (tuple): Physical voxel spacing in mm
        threshold_label (float): Binarization threshold for ground truth
        threshold_output (float): Binarization threshold for prediction
        apply_rotation (bool): Whether to apply rotation to output for alignment
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Apply rotation if needed (for alignment)
    if apply_rotation:
        output = ndimage.rotate(output, angle=-90, axes=(2, 1), reshape=True, order=0)  # [ORIGINAL] kept. Reason: aligns the prediction orientation with the ground truth.
    
    # [EDIT] Remove tiny disconnected fragments from the prediction before computing any metric.
    # Reason: NeCA's evaluation protocol (paper Sec. 2.5) does this with a 25-voxel limit; eval_orig.py did not.
    if min_component_voxels > 0:
        output = remove_small_components(output, threshold_output, min_component_voxels)
    
    # Compute all metrics
    metrics = {}
    
    try:
        metrics['dice'] = compute_overlap_metric(
            label, output, d=0.0, 
            threshold_output=threshold_output, 
            threshold_label=threshold_label, 
            voxel_spacing=voxel_spacing
        )
        
        metrics['iou'] = compute_iou(
            label, output,
            threshold_label=threshold_label,
            threshold_output=threshold_output
        )
        
        metrics['cldice'] = compute_cldice(
            label, output,
            threshold_label=threshold_label,
            threshold_output=threshold_output
        )
        
        metrics['chamfer_distance'] = compute_chamfer_distance(
            label, output, 
            physical_units=True, 
            voxel_spacing=voxel_spacing, 
            threshold_output=threshold_output, 
            threshold_label=threshold_label
        )
        
        metrics['reconstruction_error'] = compute_reconstruction_error(label, output)
        
        metrics['remse'] = compute_remse(label, output)
        
        # Add additional useful metrics
        label_bin = (label >= threshold_label).astype(np.uint8)
        output_bin = (output >= threshold_output).astype(np.uint8)
        
        metrics['label_volume'] = np.sum(label_bin) * np.prod(voxel_spacing)  # mm³
        metrics['output_volume'] = np.sum(output_bin) * np.prod(voxel_spacing)  # mm³
        metrics['volume_ratio'] = metrics['output_volume'] / metrics['label_volume'] if metrics['label_volume'] > 0 else np.inf
        
    except Exception as e:
        print(f"Error computing metrics: {str(e)}")
        metrics = {
            'dice': np.nan,
            'iou': np.nan,
            'cldice': np.nan,
            'chamfer_distance': np.nan,
            'reconstruction_error': np.nan,
            'remse': np.nan,
            'label_volume': np.nan,
            'output_volume': np.nan,
            'volume_ratio': np.nan
        }
    
    return metrics


def evaluate_single_model(
    model_id: int,
    gt_volume_path: str,
    recon_volume_path: str,
    voxel_spacing: tuple = (0.8, 0.8, 0.8),
    threshold_label: float = 0.5,   # [EDIT] was 0.8. Reason: NeCA binarises at 0.5 (see compute_all_metrics).
    threshold_output: float = 0.5,  # [EDIT] was 0.8. Reason: same as above.
    apply_rotation: bool = True,
    min_component_voxels: int = 25  # [EDIT] new argument, passed to compute_all_metrics.
) -> dict:
    """
    Evaluate a single model by comparing ground truth and reconstruction.
    
    Args:
        model_id (int): Model identifier
        gt_volume_path (str): Path to ground truth volume
        recon_volume_path (str): Path to reconstructed volume
        voxel_spacing (tuple): Physical voxel spacing in mm
        threshold_label (float): Binarization threshold for ground truth
        threshold_output (float): Binarization threshold for prediction
        apply_rotation (bool): Whether to apply rotation for alignment
    
    Returns:
        dict: Evaluation results for this model
    """
    
    print(f"Evaluating Model {model_id}")
    print(f"  GT: {gt_volume_path}")
    print(f"  Recon: {recon_volume_path}")
    
    try:
        # Load volumes
        if not os.path.exists(gt_volume_path):
            raise FileNotFoundError(f"Ground truth not found: {gt_volume_path}")
        if not os.path.exists(recon_volume_path):
            raise FileNotFoundError(f"Reconstruction not found: {recon_volume_path}")
        
        label = np.load(gt_volume_path).astype(np.float32)
        output = np.load(recon_volume_path).astype(np.float32)
        
        print(f"  GT shape: {label.shape}, Recon shape: {output.shape}")
        
        # Compute metrics
        metrics = compute_all_metrics(
            label, output, 
            voxel_spacing=voxel_spacing,
            threshold_label=threshold_label,
            threshold_output=threshold_output,
            apply_rotation=apply_rotation,
            min_component_voxels=min_component_voxels  # [EDIT] new argument, see compute_all_metrics.
        )
        
        # Add metadata
        result = {
            'model_id': model_id,
            'gt_path': gt_volume_path,
            'recon_path': recon_volume_path,
            'gt_shape': label.shape,
            'recon_shape': output.shape,
            'voxel_spacing': voxel_spacing,
            'threshold_label': threshold_label,
            'threshold_output': threshold_output,
            'evaluation_success': True,
            **metrics
        }

        print(f"  ✅ Success: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, clDice={metrics['cldice']:.4f}, Chamfer={metrics['chamfer_distance']:.4f}mm, RError={metrics['reconstruction_error']:.4f}, reMSE={metrics['remse']:.4f}")

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        result = {
            'model_id': model_id,
            'gt_path': gt_volume_path,
            'recon_path': recon_volume_path,
            'evaluation_success': False,
            'error_message': str(e),
            'dice': np.nan,
            'iou': np.nan,
            'cldice': np.nan,
            'chamfer_distance': np.nan,
            'reconstruction_error': np.nan,
            'remse': np.nan,
            'label_volume': np.nan,
            'output_volume': np.nan,
            'volume_ratio': np.nan
        }
    
    return result


def batch_evaluate_models(
    config_path: str = "./config/CCTA.yaml",
    model_numbers: list = None,
    voxel_spacing: tuple = (0.8, 0.8, 0.8),
    threshold_label: float = 0.5,   # [EDIT] was 0.8. Reason: NeCA binarises at 0.5 (see compute_all_metrics).
    threshold_output: float = 0.5,  # [EDIT] was 0.8. Reason: same as above.
    apply_rotation: bool = True,
    output_dir: str = "./logs/evaluation/",
    min_component_voxels: int = 25  # [EDIT] new argument, passed down to compute_all_metrics.
) -> pd.DataFrame:
    """
    Batch evaluate multiple models and save comprehensive results.
    
    Args:
        config_path (str): Path to configuration file
        model_numbers (list): List of model IDs to evaluate (None = use config)
        voxel_spacing (tuple): Physical voxel spacing in mm
        threshold_label (float): Binarization threshold for ground truth
        threshold_output (float): Binarization threshold for prediction
        apply_rotation (bool): Whether to apply rotation for alignment
        output_dir (str): Directory to save evaluation results
    
    Returns:
        pd.DataFrame: DataFrame containing all evaluation results
    """
    
    print("🔍 BATCH EVALUATION OF NECA MODELS")
    print("=" * 60)
    
    # Load configuration
    try:
        cfg = load_config(config_path)
        print(f"✅ Loaded configuration: {config_path}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return pd.DataFrame()
    
    # Get paths from config
    input_data_dir = cfg["exp"].get("input_data_dir", "./data/GT_volumes/")
    output_recon_dir = cfg["exp"].get("output_recon_dir", "./logs/reconstructions/")
    
    print(f"Input GT directory: {input_data_dir}")
    print(f"Reconstructions directory: {output_recon_dir}")
    print(f"Scanning for recon_occupancy_*.npy files...")
    print(f"Evaluation parameters:")
    print(f"  Voxel spacing: {voxel_spacing} mm")
    print(f"  GT threshold: {threshold_label}")
    print(f"  Prediction threshold: {threshold_output}")
    print(f"  Apply rotation: {apply_rotation}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover all reconstruction files automatically
    recon_files = []
    for root, dirs, files in os.walk(output_recon_dir):
        for file in files:
            if file.startswith("recon_occupancy_") and file.endswith(".npy"):
                full_path = os.path.join(root, file)
                # Extract experiment name from filename (remove recon_occupancy_ prefix and .npy suffix)
                experiment_name = file[16:-4]  # Remove "recon_occupancy_" (16 chars) and ".npy" (4 chars)
                recon_files.append((experiment_name, full_path))
    
    print(f"Found {len(recon_files)} reconstruction files")
    
    # Evaluate each found reconstruction
    results = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    for experiment_name, recon_path in recon_files:
        print(f"\n📊 Evaluating {experiment_name}")
        print("-" * 40)
        
        # Extract base model ID from experiment name (first part before _lr)
        try:
            base_model_id = experiment_name.split('_lr')[0]
            gt_path = os.path.join(input_data_dir, f"{base_model_id}.npy")
        except:
            print(f"  ⚠️  Cannot extract model ID from {experiment_name}, skipping")
            continue
        
        print(f"  Base model: {base_model_id}")
        print(f"  Reconstruction: {recon_path}")
        
        # Evaluate single experiment
        result = evaluate_single_model(
            model_id=experiment_name,  # Use experiment name as identifier
            gt_volume_path=gt_path,
            recon_volume_path=recon_path,
            voxel_spacing=voxel_spacing,
            threshold_label=threshold_label,
            threshold_output=threshold_output,
            apply_rotation=apply_rotation,
            min_component_voxels=min_component_voxels  # [EDIT] new argument, see compute_all_metrics.
        )
        
        # Add experiment metadata (parse from experiment name if possible)
        result['base_model_id'] = base_model_id
        result['experiment_name'] = experiment_name
        
        # Try to parse experiment parameters from filename
        try:
            parts = experiment_name.split('_')
            for part in parts:
                if part.startswith('lr'):
                    result['learning_rate'] = float(part[2:])
                elif part.startswith('proj'):
                    result['projection_weight'] = float(part[4:])
                elif part.startswith('sdf'):
                    result['sdf_weight'] = float(part[3:])
        except:
            # If parsing fails, set defaults
            result['learning_rate'] = None
            result['projection_weight'] = None  
            result['sdf_weight'] = None
        
        results.append(result)
        
        if result['evaluation_success']:
            successful_evaluations += 1
        else:
            failed_evaluations += 1
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results in multiple formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV for easy viewing/analysis
    csv_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Saved CSV results: {csv_path}")
    
    # JSON for programmatic access
    json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    df_results.to_json(json_path, orient='records', indent=2)
    print(f"💾 Saved JSON results: {json_path}")
    
    # Summary statistics - only for successful evaluations
    if successful_evaluations > 0:
        # Filter only successful evaluations for statistics
        successful_results = df_results[df_results['evaluation_success'] == True]
        
        summary_stats = {
            'timestamp': timestamp,
            'total_experiments': len(recon_files),
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': failed_evaluations,
            'parameters': {
                'voxel_spacing': voxel_spacing,
                'threshold_label': threshold_label,
                'threshold_output': threshold_output,
                'apply_rotation': apply_rotation,
                'min_component_voxels': min_component_voxels  # [EDIT] logged so each run records the cleanup setting.
            },
            'statistics': {
                'dice_mean': float(successful_results['dice'].mean()),
                'dice_std': float(successful_results['dice'].std()),
                'iou_mean': float(successful_results['iou'].mean()),
                'iou_std': float(successful_results['iou'].std()),
                'cldice_mean': float(successful_results['cldice'].mean()),
                'cldice_std': float(successful_results['cldice'].std()),
                'chamfer_distance_mean': float(successful_results['chamfer_distance'].mean()),
                'chamfer_distance_std': float(successful_results['chamfer_distance'].std()),
                'reconstruction_error_mean': float(successful_results['reconstruction_error'].mean()),
                'reconstruction_error_std': float(successful_results['reconstruction_error'].std()),
                'remse_mean': float(successful_results['remse'].mean()),
                'remse_std': float(successful_results['remse'].std())
            }
        }
        
        summary_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"📈 Saved summary statistics: {summary_path}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total experiments: {len(recon_files)}")
    print(f"Successful: {successful_evaluations}")
    print(f"Failed: {failed_evaluations}")
    
    if successful_evaluations > 0:
        # Filter only successful evaluations for statistics display
        successful_results = df_results[df_results['evaluation_success'] == True]
        
        print(f"\n📊 Performance Statistics (based on {successful_evaluations} successful evaluations):")
        print(f"clDice (%):       {successful_results['cldice'].mean()*100:.2f} ± {successful_results['cldice'].std()*100:.2f}")
        print(f"Dice (%):         {successful_results['dice'].mean()*100:.2f} ± {successful_results['dice'].std()*100:.2f}")
        print(f"IoU (%):          {successful_results['iou'].mean()*100:.2f} ± {successful_results['iou'].std()*100:.2f}")
        print(f"reError:          {successful_results['reconstruction_error'].mean():.2f} ± {successful_results['reconstruction_error'].std():.2f}")
        print(f"CD_l2 (mm):       {successful_results['chamfer_distance'].mean():.2f} ± {successful_results['chamfer_distance'].std():.2f}")
        print(f"reMSE (×1e-4):    {successful_results['remse'].mean()*1e4:.2f} ± {successful_results['remse'].std()*1e4:.2f}")
    
    if failed_evaluations > 0:
        print(f"\n❌ Failed models: {df_results[~df_results['evaluation_success']]['model_id'].tolist()}")
    
    return df_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate NeCA reconstructions against ground truth")
    
    parser.add_argument("--config", default="./config/CCTA.yaml",
                       help="Path to configuration file")
    parser.add_argument("--models", nargs='+', type=int, default=None,
                       help="Specific model IDs to evaluate (default: use config)")
    parser.add_argument("--single", type=int, default=None,
                       help="Evaluate only a single model")
    parser.add_argument("--output-dir", default="./logs/evaluation/",
                       help="Directory to save evaluation results")
    # [EDIT] Both threshold defaults changed from 0.8 to 0.5. Reason: NeCA binarises at 0.5 (see compute_all_metrics).
    parser.add_argument("--threshold-gt", type=float, default=0.5,
                       help="Binarization threshold for ground truth")
    parser.add_argument("--threshold-pred", type=float, default=0.5,
                       help="Binarization threshold for predictions")
    # [EDIT] New option. Reason: lets you switch NeCA's small-fragment cleanup off (0) to see its effect.
    parser.add_argument("--min-component-voxels", type=int, default=25,
                       help="Remove predicted components smaller than this many voxels (0 = off)")
    parser.add_argument("--no-rotation", action='store_true',
                       help="Skip rotation alignment")
    parser.add_argument("--voxel-spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8],
                       help="Voxel spacing in mm (x y z)")
    
    args = parser.parse_args()
    
    # Handle single model evaluation
    if args.single is not None:
        args.models = [args.single]
    
    # Run batch evaluation
    results_df = batch_evaluate_models(
        config_path=args.config,
        model_numbers=args.models,
        voxel_spacing=tuple(args.voxel_spacing),
        threshold_label=args.threshold_gt,
        threshold_output=args.threshold_pred,
        apply_rotation=not args.no_rotation,
        output_dir=args.output_dir,
        min_component_voxels=args.min_component_voxels  # [EDIT] new argument, see compute_all_metrics.
    )
    
    return results_df


if __name__ == "__main__":
    main()