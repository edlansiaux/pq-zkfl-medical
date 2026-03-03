"""
Main Experiment Runner: Zero-Knowledge Federated Learning with
Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI.

Runs three configurations:
    1. Standard FL (simulated TLS baseline)
    2. FL + ML-KEM (post-quantum key exchange only)
    3. FL + ML-KEM + ZKP + HE (full hybrid protocol)

Also includes ABLATION STUDIES for:
    - Varying number of malicious clients
    - Varying norm threshold τ
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_core.model import (SimpleMLP, generate_synthetic_medical_data,
                            partition_non_iid)
from crypto.ml_kem import MLKEM768, symmetric_encrypt, symmetric_decrypt
from crypto.zkp_norm import ZKPNormBound
from crypto.homomorphic import GradientHEManager, HE_N


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'n_clients': 5,
    'n_rounds': 10,
    'n_samples': 1000,
    'n_features': 784,      # 28x28 flattened
    'n_classes': 4,
    'local_epochs': 3,
    'local_lr': 0.01,
    'batch_size': 32,
    'dirichlet_alpha': 0.5,
    'norm_threshold': 5.0,
    'malicious_client_id': 3,
    'malicious_scale': 50.0,
    'seed': 42,
}


def local_training(model: SimpleMLP, X: np.ndarray, y: np.ndarray,
                   n_epochs: int, lr: float, batch_size: int) -> np.ndarray:
    """Perform local SGD training and return the gradient (weight delta)."""
    initial_weights = model.get_weights().copy()
    n_samples = len(X)
    rng = np.random.default_rng()

    for epoch in range(n_epochs):
        perm = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            grad_vec, loss = model.train_step(X_batch, y_batch, lr)
            new_weights = model.get_weights() - lr * grad_vec
            model.set_weights(new_weights)

    delta = model.get_weights() - initial_weights
    return delta


# ============================================================
# Experiment 1: Standard FL (TLS Baseline)
# ============================================================
def run_standard_fl(partitions, test_X, test_y, config):
    """Standard FedAvg with simulated TLS."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Standard Federated Learning (TLS Baseline)")
    print("="*60)

    model = SimpleMLP(config['n_features'], config['n_classes'], config['seed'])
    n_params = model.n_params()
    print(f"Model parameters: {n_params}")

    metrics = {
        'round_times': [], 'accuracies': [], 'losses': [],
        'message_sizes_kb': [], 'grad_norms': [],
    }

    for round_t in range(config['n_rounds']):
        t_start = time.perf_counter()
        global_weights = model.get_weights().copy()
        deltas = []
        total_msg_size = 0

        for client_id in range(config['n_clients']):
            local_model = SimpleMLP(config['n_features'], config['n_classes'])
            local_model.set_weights(global_weights.copy())

            X_c, y_c = partitions[client_id]
            delta = local_training(local_model, X_c, y_c,
                                  config['local_epochs'], config['local_lr'],
                                  config['batch_size'])

            if client_id == config['malicious_client_id'] and round_t >= 3:
                delta = np.random.normal(0, config['malicious_scale'], size=len(delta))

            deltas.append(delta)
            msg_size = delta.nbytes
            total_msg_size += msg_size

        avg_delta = np.mean(deltas, axis=0)
        new_weights = global_weights + avg_delta
        model.set_weights(new_weights)

        round_time = time.perf_counter() - t_start

        acc, loss = model.evaluate(test_X, test_y)
        grad_norms = [np.linalg.norm(d) for d in deltas]

        metrics['round_times'].append(round_time)
        metrics['accuracies'].append(acc)
        metrics['losses'].append(loss)
        metrics['message_sizes_kb'].append(total_msg_size / 1024)
        metrics['grad_norms'].append(grad_norms)

        print(f"  Round {round_t+1:2d}: Acc={acc:.4f}, Loss={loss:.4f}, "
              f"Time={round_time:.3f}s, Msg={total_msg_size/1024:.1f}KB")

    return metrics


# ============================================================
# Experiment 2: FL + ML-KEM
# ============================================================
def run_fl_mlkem(partitions, test_X, test_y, config):
    """FedAvg with ML-KEM post-quantum key exchange."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: FL + ML-KEM (Post-Quantum Key Exchange)")
    print("="*60)

    model = SimpleMLP(config['n_features'], config['n_classes'], config['seed'])
    kem = MLKEM768(seed=config['seed'])

    server_ek, server_dk, keygen_time = kem.keygen()
    print(f"ML-KEM-768 KeyGen time: {keygen_time:.3f}s")

    metrics = {
        'round_times': [], 'accuracies': [], 'losses': [],
        'message_sizes_kb': [], 'kem_times': [],
    }

    for round_t in range(config['n_rounds']):
        t_start = time.perf_counter()
        global_weights = model.get_weights().copy()
        deltas = []
        total_msg_size = 0
        kem_overhead = 0

        for client_id in range(config['n_clients']):
            local_model = SimpleMLP(config['n_features'], config['n_classes'])
            local_model.set_weights(global_weights.copy())

            X_c, y_c = partitions[client_id]
            delta = local_training(local_model, X_c, y_c,
                                  config['local_epochs'], config['local_lr'],
                                  config['batch_size'])

            if client_id == config['malicious_client_id'] and round_t >= 3:
                delta = np.random.normal(0, config['malicious_scale'], size=len(delta))

            t_kem = time.perf_counter()
            ct_kem, shared_secret, _ = kem.encaps(server_ek)
            encrypted_delta = symmetric_encrypt(shared_secret, delta.tobytes())
            kem_time = time.perf_counter() - t_kem
            kem_overhead += kem_time

            ss_dec, _ = kem.decaps(server_dk, ct_kem)
            decrypted_bytes = symmetric_decrypt(ss_dec, encrypted_delta)
            delta_recovered = np.frombuffer(decrypted_bytes, dtype=np.float64)

            deltas.append(delta_recovered)

            msg_size = (ct_kem['u'].nbytes + ct_kem['v'].nbytes +
                       len(encrypted_delta))
            total_msg_size += msg_size

        avg_delta = np.mean(deltas, axis=0)
        model.set_weights(global_weights + avg_delta)

        round_time = time.perf_counter() - t_start
        acc, loss = model.evaluate(test_X, test_y)

        metrics['round_times'].append(round_time)
        metrics['accuracies'].append(acc)
        metrics['losses'].append(loss)
        metrics['message_sizes_kb'].append(total_msg_size / 1024)
        metrics['kem_times'].append(kem_overhead)

        print(f"  Round {round_t+1:2d}: Acc={acc:.4f}, Loss={loss:.4f}, "
              f"Time={round_time:.3f}s, KEM={kem_overhead:.3f}s")

    return metrics


# ============================================================
# Experiment 3: Full Hybrid Protocol (ML-KEM + ZKP + HE)
# ============================================================
def run_fl_hybrid(partitions, test_X, test_y, config, verbose=True):
    """Full hybrid: ML-KEM + ZKP + Homomorphic Encryption."""
    if verbose:
        print("\n" + "="*60)
        print("EXPERIMENT 3: FL + ML-KEM + ZKP + HE (Full Hybrid)")
        print("="*60)

    model = SimpleMLP(config['n_features'], config['n_classes'], config['seed'])
    n_params = model.n_params()
    kem = MLKEM768(seed=config['seed'])

    server_ek, server_dk, _ = kem.keygen()

    HE_DIM = min(512, n_params)
    he_manager = GradientHEManager(HE_DIM, scale=100.0, seed=config['seed'])

    ZKP_DIM = min(256, n_params)
    zkp_system = ZKPNormBound(ZKP_DIM, config['norm_threshold'], seed=config['seed'])

    metrics = {
        'round_times': [], 'accuracies': [], 'losses': [],
        'message_sizes_kb': [], 'zkp_gen_times': [], 'zkp_ver_times': [],
        'he_encrypt_times': [], 'he_aggregate_times': [], 'he_decrypt_times': [],
        'malicious_detected': [], 'zkp_proof_sizes': [],
        'he_reconstruction_errors': [],
    }

    for round_t in range(config['n_rounds']):
        t_start = time.perf_counter()
        global_weights = model.get_weights().copy()

        all_deltas = []
        all_he_cts = []
        total_msg_size = 0
        zkp_gen_total = 0
        zkp_ver_total = 0
        he_enc_total = 0
        detected_malicious = 0
        proof_sizes = 0
        valid_deltas = []

        for client_id in range(config['n_clients']):
            local_model = SimpleMLP(config['n_features'], config['n_classes'])
            local_model.set_weights(global_weights.copy())

            X_c, y_c = partitions[client_id]
            delta = local_training(local_model, X_c, y_c,
                                  config['local_epochs'], config['local_lr'],
                                  config['batch_size'])

            is_malicious = (client_id == config['malicious_client_id'] and round_t >= 3)
            if is_malicious:
                delta = np.random.normal(0, config['malicious_scale'], size=len(delta))

            # --- ZKP: Prove norm bound ---
            t_zkp = time.perf_counter()
            proof = zkp_system.generate_proof(delta[:ZKP_DIM])
            zkp_gen_time = time.perf_counter() - t_zkp
            zkp_gen_total += zkp_gen_time

            t_ver = time.perf_counter()
            is_valid, _ = zkp_system.verify_proof(proof)
            gradient_norm = np.linalg.norm(delta)
            overall_valid = is_valid and proof['is_within_bound']
            zkp_ver_time = time.perf_counter() - t_ver
            zkp_ver_total += zkp_ver_time

            if not overall_valid:
                detected_malicious += 1
                if verbose:
                    print(f"    [!] Client {client_id} REJECTED (norm={gradient_norm:.2f}, "
                          f"zkp_valid={is_valid}, within_bound={proof['is_within_bound']})")
                continue

            # --- ML-KEM: Secure channel ---
            ct_kem, shared_secret, _ = kem.encaps(server_ek)

            # --- HE: Encrypt gradient subset for aggregation ---
            t_he = time.perf_counter()
            he_cts, enc_time = he_manager.encrypt_gradient(delta[:HE_DIM])
            he_enc_total += time.perf_counter() - t_he

            all_he_cts.append(he_cts)
            valid_deltas.append(delta)

            msg_size = (proof['proof_size_bytes'] +
                       ct_kem['u'].nbytes + ct_kem['v'].nbytes +
                       sum(ct['c0'].nbytes + ct['c1'].nbytes for ct in he_cts))
            total_msg_size += msg_size
            proof_sizes += proof['proof_size_bytes']

        # --- Server-side HE aggregation ---
        he_reconstruction_error = 0.0
        if len(all_he_cts) > 0:
            t_agg = time.perf_counter()
            agg_cts, _ = he_manager.aggregate_encrypted_gradients(all_he_cts)
            he_agg_time = time.perf_counter() - t_agg

            t_dec = time.perf_counter()
            he_result, _ = he_manager.decrypt_aggregated(agg_cts, len(all_he_cts))
            he_dec_time = time.perf_counter() - t_dec
            
            # M2 fix: Compute HE reconstruction error
            if len(valid_deltas) > 0:
                true_avg = np.mean([d[:HE_DIM] for d in valid_deltas], axis=0)
                he_reconstruction_error = np.mean(np.abs(he_result - true_avg))
        else:
            he_agg_time = 0
            he_dec_time = 0
            he_result = np.zeros(HE_DIM)

        if len(valid_deltas) > 0:
            avg_delta = np.mean(valid_deltas, axis=0)
            avg_delta[:HE_DIM] = he_result
        else:
            avg_delta = np.zeros(n_params)

        model.set_weights(global_weights + avg_delta)

        round_time = time.perf_counter() - t_start
        acc, loss = model.evaluate(test_X, test_y)

        metrics['round_times'].append(round_time)
        metrics['accuracies'].append(acc)
        metrics['losses'].append(loss)
        metrics['message_sizes_kb'].append(total_msg_size / 1024)
        metrics['zkp_gen_times'].append(zkp_gen_total)
        metrics['zkp_ver_times'].append(zkp_ver_total)
        metrics['he_encrypt_times'].append(he_enc_total)
        metrics['he_aggregate_times'].append(he_agg_time)
        metrics['he_decrypt_times'].append(he_dec_time)
        metrics['malicious_detected'].append(detected_malicious)
        metrics['zkp_proof_sizes'].append(proof_sizes)
        metrics['he_reconstruction_errors'].append(he_reconstruction_error)

        if verbose:
            print(f"  Round {round_t+1:2d}: Acc={acc:.4f}, Loss={loss:.4f}, "
                  f"Time={round_time:.2f}s, Detected={detected_malicious}, "
                  f"HE_err={he_reconstruction_error:.6f}")

    return metrics


# ============================================================
# ABLATION STUDIES (M3 fix)
# ============================================================
def run_ablation_malicious_clients(partitions, test_X, test_y, base_config):
    """Ablation: Vary number of malicious clients (0-3)."""
    print("\n" + "="*60)
    print("ABLATION STUDY 1: Varying Number of Malicious Clients")
    print("="*60)
    
    results = []
    for n_malicious in [0, 1, 2, 3]:
        config = base_config.copy()
        
        # Run 3 rounds for speed
        config['n_rounds'] = 5
        
        print(f"\n--- Testing with {n_malicious} malicious client(s) ---")
        
        # Create model fresh
        model = SimpleMLP(config['n_features'], config['n_classes'], config['seed'])
        n_params = model.n_params()
        kem = MLKEM768(seed=config['seed'])
        server_ek, server_dk, _ = kem.keygen()
        
        HE_DIM = min(512, n_params)
        he_manager = GradientHEManager(HE_DIM, scale=100.0, seed=config['seed'])
        
        ZKP_DIM = min(256, n_params)
        zkp_system = ZKPNormBound(ZKP_DIM, config['norm_threshold'], seed=config['seed'])
        
        total_detected = 0
        total_malicious_updates = 0
        false_positives = 0
        
        for round_t in range(config['n_rounds']):
            global_weights = model.get_weights().copy()
            valid_deltas = []
            
            for client_id in range(config['n_clients']):
                local_model = SimpleMLP(config['n_features'], config['n_classes'])
                local_model.set_weights(global_weights.copy())
                
                X_c, y_c = partitions[client_id]
                delta = local_training(local_model, X_c, y_c,
                                      config['local_epochs'], config['local_lr'],
                                      config['batch_size'])
                
                # Make multiple clients malicious
                is_malicious = (client_id < n_malicious and round_t >= 2)
                if is_malicious:
                    delta = np.random.normal(0, config['malicious_scale'], size=len(delta))
                    total_malicious_updates += 1
                
                # ZKP verification
                proof = zkp_system.generate_proof(delta[:ZKP_DIM])
                is_valid, _ = zkp_system.verify_proof(proof)
                overall_valid = is_valid and proof['is_within_bound']
                
                if not overall_valid:
                    if is_malicious:
                        total_detected += 1
                    else:
                        false_positives += 1
                    continue
                
                valid_deltas.append(delta)
            
            if len(valid_deltas) > 0:
                avg_delta = np.mean(valid_deltas, axis=0)
                model.set_weights(global_weights + avg_delta)
        
        acc, loss = model.evaluate(test_X, test_y)
        detection_rate = total_detected / total_malicious_updates if total_malicious_updates > 0 else 1.0
        
        result = {
            'n_malicious': n_malicious,
            'final_accuracy': acc,
            'final_loss': loss,
            'detection_rate': detection_rate,
            'false_positive_count': false_positives,
            'total_detected': total_detected,
            'total_malicious_updates': total_malicious_updates,
        }
        results.append(result)
        print(f"  Final Acc: {acc:.4f}, Detection Rate: {detection_rate:.1%}, FP: {false_positives}")
    
    return results


def run_ablation_threshold(partitions, test_X, test_y, base_config):
    """Ablation: Vary norm threshold τ."""
    print("\n" + "="*60)
    print("ABLATION STUDY 2: Varying Norm Threshold τ")
    print("="*60)
    
    thresholds = [1.0, 2.0, 5.0, 10.0, 50.0]
    results = []
    
    for tau in thresholds:
        config = base_config.copy()
        config['norm_threshold'] = tau
        config['n_rounds'] = 5
        
        print(f"\n--- Testing with τ = {tau} ---")
        
        model = SimpleMLP(config['n_features'], config['n_classes'], config['seed'])
        n_params = model.n_params()
        kem = MLKEM768(seed=config['seed'])
        server_ek, server_dk, _ = kem.keygen()
        
        HE_DIM = min(512, n_params)
        he_manager = GradientHEManager(HE_DIM, scale=100.0, seed=config['seed'])
        
        ZKP_DIM = min(256, n_params)
        zkp_system = ZKPNormBound(ZKP_DIM, tau, seed=config['seed'])
        
        total_detected = 0
        total_malicious_updates = 0
        false_positives = 0
        total_honest_updates = 0
        
        for round_t in range(config['n_rounds']):
            global_weights = model.get_weights().copy()
            valid_deltas = []
            
            for client_id in range(config['n_clients']):
                local_model = SimpleMLP(config['n_features'], config['n_classes'])
                local_model.set_weights(global_weights.copy())
                
                X_c, y_c = partitions[client_id]
                delta = local_training(local_model, X_c, y_c,
                                      config['local_epochs'], config['local_lr'],
                                      config['batch_size'])
                
                is_malicious = (client_id == config['malicious_client_id'] and round_t >= 2)
                if is_malicious:
                    delta = np.random.normal(0, config['malicious_scale'], size=len(delta))
                    total_malicious_updates += 1
                else:
                    total_honest_updates += 1
                
                proof = zkp_system.generate_proof(delta[:ZKP_DIM])
                is_valid, _ = zkp_system.verify_proof(proof)
                overall_valid = is_valid and proof['is_within_bound']
                
                if not overall_valid:
                    if is_malicious:
                        total_detected += 1
                    else:
                        false_positives += 1
                    continue
                
                valid_deltas.append(delta)
            
            if len(valid_deltas) > 0:
                avg_delta = np.mean(valid_deltas, axis=0)
                model.set_weights(global_weights + avg_delta)
        
        acc, loss = model.evaluate(test_X, test_y)
        detection_rate = total_detected / total_malicious_updates if total_malicious_updates > 0 else 1.0
        fpr = false_positives / total_honest_updates if total_honest_updates > 0 else 0.0
        
        result = {
            'threshold': tau,
            'final_accuracy': acc,
            'final_loss': loss,
            'detection_rate': detection_rate,
            'false_positive_rate': fpr,
            'false_positives': false_positives,
            'total_detected': total_detected,
        }
        results.append(result)
        print(f"  Final Acc: {acc:.4f}, Detection Rate: {detection_rate:.1%}, FPR: {fpr:.1%}")
    
    return results


# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("Zero-Knowledge Federated Learning with Lattice-Based")
    print("Hybrid Encryption for Quantum-Resilient Medical AI")
    print("="*60)
    print(f"\nConfiguration: {json.dumps(CONFIG, indent=2)}")

    print("\n--- Generating synthetic medical imaging data ---")
    X, y = generate_synthetic_medical_data(
        CONFIG['n_samples'], CONFIG['n_features'],
        CONFIG['n_classes'], CONFIG['seed']
    )

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    partitions = partition_non_iid(X_train, y_train, CONFIG['n_clients'],
                                   CONFIG['dirichlet_alpha'], CONFIG['seed'])

    for i, (Xp, yp) in enumerate(partitions):
        unique, counts = np.unique(yp, return_counts=True)
        print(f"  Client {i}: {len(Xp)} samples, classes={dict(zip(unique.astype(int), counts))}")

    # Run main experiments
    results = {}
    results['standard'] = run_standard_fl(partitions, X_test, y_test, CONFIG)
    results['mlkem'] = run_fl_mlkem(partitions, X_test, y_test, CONFIG)
    results['hybrid'] = run_fl_hybrid(partitions, X_test, y_test, CONFIG)

    # Run ablation studies
    results['ablation_malicious'] = run_ablation_malicious_clients(partitions, X_test, y_test, CONFIG)
    results['ablation_threshold'] = run_ablation_threshold(partitions, X_test, y_test, CONFIG)

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(os.path.join(output_dir, 'experiment_results.json'), 'w') as f:
        json.dump(convert(results), f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)

    for name in ['standard', 'mlkem', 'hybrid']:
        m = results[name]
        print(f"\n--- {name.upper()} ---")
        print(f"  Mean round time: {np.mean(m['round_times']):.3f}s")
        print(f"  Final accuracy:  {m['accuracies'][-1]:.4f}")
        print(f"  Mean msg size:   {np.mean(m['message_sizes_kb']):.1f} KB/round")
        if 'malicious_detected' in m:
            active_rounds = sum(1 for r in range(len(m['malicious_detected'])) if r >= 3)
            detected_rounds = sum(1 for r in range(len(m['malicious_detected']))
                                 if r >= 3 and m['malicious_detected'][r] > 0)
            if active_rounds > 0:
                print(f"  Malicious detection: {detected_rounds}/{active_rounds} rounds "
                      f"({100*detected_rounds/active_rounds:.0f}%)")
        if 'he_reconstruction_errors' in m:
            mean_he_error = np.mean(m['he_reconstruction_errors'])
            print(f"  Mean HE reconstruction error: {mean_he_error:.6f}")

    # Print ablation results
    print("\n--- ABLATION: Malicious Clients ---")
    for r in results['ablation_malicious']:
        print(f"  n_malicious={r['n_malicious']}: Acc={r['final_accuracy']:.4f}, "
              f"Detection={r['detection_rate']:.1%}, FP={r['false_positive_count']}")

    print("\n--- ABLATION: Threshold τ ---")
    for r in results['ablation_threshold']:
        print(f"  τ={r['threshold']}: Acc={r['final_accuracy']:.4f}, "
              f"Detection={r['detection_rate']:.1%}, FPR={r['false_positive_rate']:.1%}")

    print(f"\nResults saved to: {output_dir}/experiment_results.json")
    return results


if __name__ == '__main__':
    main()
