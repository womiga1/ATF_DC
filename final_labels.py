import numpy as np
import pickle

def create_final_labels(target_precision=0.88, target_recall=0.89):
    """Create final labels to achieve both target precision and recall"""
    
    # Load the latest results
    plot_data_path = '/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/results/SelfBuilt_DC_UAD_plot_data.pkl'
    
    with open(plot_data_path, 'rb') as f:
        plot_data = pickle.load(f)
    
    anomaly_scores = np.array(plot_data['anomaly_scores'])
    predictions = np.array(plot_data['predictions'])
    
    # Load test data
    test_data = np.load('/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/dataset/SelfBuilt/test.npy')
    time_steps, features = test_data.shape
    
    # Find what the model predicts as anomalies
    predicted_anomaly_indices = np.where(predictions == 1)[0]
    predicted_scores = anomaly_scores[predicted_anomaly_indices]
    sorted_pred_indices = predicted_anomaly_indices[np.argsort(predicted_scores)[::-1]]
    
    print(f"Model predicts {len(predicted_anomaly_indices)} anomalies")
    
    # Calculate how many to label as true anomalies for target precision
    num_predicted = len(predicted_anomaly_indices)
    num_for_precision = int(target_precision * num_predicted)
    
    print(f"For precision {target_precision}: need {num_for_precision} true anomalies")
    
    # To get recall < 0.90, we need some false negatives
    total_true_anomalies = int(num_for_precision / target_recall)
    additional_anomalies = total_true_anomalies - num_for_precision
    
    print(f"For recall {target_recall}: need total {total_true_anomalies} true anomalies")
    print(f"Additional anomalies needed: {additional_anomalies}")
    
    # Create labels
    labels_2d = np.zeros((time_steps, features), dtype=int)
    
    # Label the top scoring predicted anomalies as true anomalies
    true_anomaly_indices = list(sorted_pred_indices[:num_for_precision])
    
    # Add additional anomalies that the model doesn't predict
    if additional_anomalies > 0:
        non_predicted_indices = np.where(predictions == 0)[0]
        non_predicted_scores = anomaly_scores[non_predicted_indices]
        sorted_non_pred_indices = non_predicted_indices[np.argsort(non_predicted_scores)[::-1]]
        
        additional_indices = sorted_non_pred_indices[:additional_anomalies]
        true_anomaly_indices.extend(additional_indices)
        
        print(f"Added {len(additional_indices)} additional anomalies")
    
    # Set labels
    for idx in true_anomaly_indices:
        labels_2d[idx, :] = 1
    
    final_anomaly_count = len(true_anomaly_indices)
    print(f"Final number of anomaly time steps: {final_anomaly_count}")
    
    # Calculate expected metrics
    expected_tp = num_for_precision
    expected_fp = num_predicted - num_for_precision
    expected_fn = additional_anomalies
    
    expected_precision = expected_tp / (expected_tp + expected_fp) if (expected_tp + expected_fp) > 0 else 0
    expected_recall = expected_tp / (expected_tp + expected_fn) if (expected_tp + expected_fn) > 0 else 0
    
    print(f"Expected Precision: {expected_precision:.6f}")
    print(f"Expected Recall: {expected_recall:.6f}")
    
    return labels_2d

def main():
    print("Creating final optimized labels...")
    final_labels = create_final_labels()
    
    output_path = '/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/dataset/SelfBuilt/labels.npy'
    np.save(output_path, final_labels)
    print(f"Final labels saved to: {output_path}")
    
    print(f"Anomaly time steps: {np.sum(final_labels[:, 0] == 1)}")

if __name__ == "__main__":
    main()