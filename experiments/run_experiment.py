"""
Main Experiment Runner: Zero-Knowledge Federated Learning with
Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI.

Runs three configurations:
    1. Standard FL (simulated TLS baseline)
    2. FL + ML-KEM (post-quantum key exchange only)
    3. FL + ML-KEM + ZKP + HE (full hybrid protocol)

Also evaluates malicious client detection and Byzantine robustness.
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
from crypto.homomorphic import GradientHEManager


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
    'dirichlet_alpha': 0.5,  # Non-IID degree
    'norm_threshold': 5.0,   # ZKP norm bound
    'malicious_client_id': 3,  # One malicious client
    'malicious_scale': 50.0,   # Scale factor for malicious gradients
    'seed': 42,
}


def local_training(model: SimpleMLP, X: np.ndarray, y: np.ndarray,
                   n_epochs: int, lr: float, batch_size: int) -> np.ndarray:
    """
    Perform local SGD training and return the gradient (weight delta).
    """
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

            # SGD update
            new_weights = model.get_weights() - lr * grad_vec
            model.set_weights(new_weights)

    # Delta = trained_weights - initial_weights
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
            # Create local model copy
            local_model = SimpleMLP(config['n_features'], config['n_classes'])
            local_model.set_weights(global_weights.copy())

            X_c, y_c = partitions[client_id]
            delta = local_training(local_model, X_c, y_c,
                                  config['local_epochs'], config['local_lr'],
                                  config['batch_size'])

            # Simulate malicious client
            if client_id == config['malicious_client_id'] and round_t >= 3:
                delta = np.random.normal(0, config['malicious_scale'], size=len(delta))

            deltas.append(delta)

            # Simulated TLS: message = serialized delta
            msg_size = delta.nbytes
            total_msg_size += msg_size

        # FedAvg aggregation
        avg_delta = np.mean(deltas, axis=0)
        new_weights = global_weights + avg_delta
        model.set_weights(new_weights)

        round_time = time.perf_counter() - t_start

        # Evaluate
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

    # Server generates KEM key pair
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

            # Malicious client
            if client_id == config['malicious_client_id'] and round_t >= 3:
                delta = np.random.normal(0, config['malicious_scale'], size=len(delta))

            # ML-KEM key exchange + symmetric encryption
            t_kem = time.perf_counter()
            ct_kem, shared_secret, _ = kem.encaps(server_ek)
            encrypted_delta = symmetric_encrypt(shared_secret, delta.tobytes())
            kem_time = time.perf_counter() - t_kem
            kem_overhead += kem_time

            # Server decrypts
            ss_dec, _ = kem.decaps(server_dk, ct_kem)
            decrypted_bytes = symmetric_decrypt(ss_dec, encrypted_delta)
            delta_recovered = np.frombuffer(decrypted_bytes, dtype=np.float64)

            deltas.append(delta_recovered)

            # Message size: KEM ciphertext + encrypted payload
            msg_size = (ct_kem['u'].nbytes + ct_kem['v'].nbytes +
                       len(encrypted_delta))
            total_msg_size += msg_size

        # Aggregation (plaintext on server)
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
def run_fl_hybrid(partitions, test_X, test_y, config):
    """Full hybrid: ML-KEM + ZKP + Homomorphic Encryption."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: FL + ML-KEM + ZKP + HE (Full Hybrid)")
    print("="*60)

    model = SimpleMLP(config['n_features'], config['n_classes'], config['seed'])
    n_params = model.n_params()
    kem = MLKEM768(seed=config['seed'])

    # Server KEM keys
    server_ek, server_dk, _ = kem.keygen()

    # HE manager (uses a subset of parameters for tractability)
    # We encrypt the first HE_DIM parameters homomorphically
    HE_DIM = min(512, n_params)  # Limit for computational tractability
    he_manager = GradientHEManager(HE_DIM, scale=100.0, seed=config['seed'])

    # ZKP system
    ZKP_DIM = min(256, n_params)
    zkp_system = ZKPNormBound(ZKP_DIM, config['norm_threshold'], seed=config['seed'])

    metrics = {
        'round_times': [], 'accuracies': [], 'losses': [],
        'message_sizes_kb': [], 'zkp_gen_times': [], 'zkp_ver_times': [],
        'he_encrypt_times': [], 'he_aggregate_times': [], 'he_decrypt_times': [],
        'malicious_detected': [], 'zkp_proof_sizes': [],
    }

    for round_t in range(config['n_rounds']):
        t_start = time.perf_counter()
        global_weights = model.get_weights().copy()

        # Per-client processing
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
            # Also check the declared norm
            gradient_norm = np.linalg.norm(delta)
            norm_valid = gradient_norm <= config['norm_threshold'] * 20  # Relaxed for non-ZKP dims
            overall_valid = is_valid and proof['is_within_bound']
            zkp_ver_time = time.perf_counter() - t_ver
            zkp_ver_total += zkp_ver_time

            if not overall_valid:
                detected_malicious += 1
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

            # Message size
            msg_size = (proof['proof_size_bytes'] +
                       ct_kem['u'].nbytes + ct_kem['v'].nbytes +
                       sum(ct['c0'].nbytes + ct['c1'].nbytes for ct in he_cts))
            total_msg_size += msg_size
            proof_sizes += proof['proof_size_bytes']

        # --- Server-side HE aggregation ---
        if len(all_he_cts) > 0:
            t_agg = time.perf_counter()
            agg_cts, _ = he_manager.aggregate_encrypted_gradients(all_he_cts)
            he_agg_time = time.perf_counter() - t_agg

            t_dec = time.perf_counter()
            he_result, _ = he_manager.decrypt_aggregated(agg_cts, len(all_he_cts))
            he_dec_time = time.perf_counter() - t_dec
        else:
            he_agg_time = 0
            he_dec_time = 0
            he_result = np.zeros(HE_DIM)

        # Combine: HE result for first HE_DIM params, plaintext avg for rest
        if len(valid_deltas) > 0:
            avg_delta = np.mean(valid_deltas, axis=0)
            # Use HE result for the encrypted portion
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

        print(f"  Round {round_t+1:2d}: Acc={acc:.4f}, Loss={loss:.4f}, "
              f"Time={round_time:.2f}s, Detected={detected_malicious}, "
              f"ZKP_gen={zkp_gen_total:.2f}s, HE_enc={he_enc_total:.2f}s")

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("Zero-Knowledge Federated Learning with Lattice-Based")
    print("Hybrid Encryption for Quantum-Resilient Medical AI")
    print("="*60)
    print(f"\nConfiguration: {json.dumps(CONFIG, indent=2)}")

    # Generate synthetic data
    print("\n--- Generating synthetic medical imaging data ---")
    X, y = generate_synthetic_medical_data(
        CONFIG['n_samples'], CONFIG['n_features'],
        CONFIG['n_classes'], CONFIG['seed']
    )

    # Split train/test (80/20)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Partition among clients (non-IID)
    partitions = partition_non_iid(X_train, y_train, CONFIG['n_clients'],
                                   CONFIG['dirichlet_alpha'], CONFIG['seed'])

    for i, (Xp, yp) in enumerate(partitions):
        unique, counts = np.unique(yp, return_counts=True)
        print(f"  Client {i}: {len(Xp)} samples, classes={dict(zip(unique.astype(int), counts))}")

    # Run experiments
    results = {}

    results['standard'] = run_standard_fl(partitions, X_test, y_test, CONFIG)
    results['mlkem'] = run_fl_mlkem(partitions, X_test, y_test, CONFIG)
    results['hybrid'] = run_fl_hybrid(partitions, X_test, y_test, CONFIG)

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy to python types for JSON
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

    for name, m in results.items():
        print(f"\n--- {name.upper()} ---")
        print(f"  Mean round time: {np.mean(m['round_times']):.3f}s")
        print(f"  Final accuracy:  {m['accuracies'][-1]:.4f}")
        print(f"  Mean msg size:   {np.mean(m['message_sizes_kb']):.1f} KB/round")
        if 'malicious_detected' in m:
            # Count rounds where malicious client was active (round >= 3)
            active_rounds = sum(1 for r in range(len(m['malicious_detected']))
                               if r >= 3)
            detected_rounds = sum(1 for r in range(len(m['malicious_detected']))
                                 if r >= 3 and m['malicious_detected'][r] > 0)
            if active_rounds > 0:
                print(f"  Malicious detection: {detected_rounds}/{active_rounds} rounds "
                      f"({100*detected_rounds/active_rounds:.0f}%)")

    print(f"\nResults saved to: {output_dir}/experiment_results.json")
    return results


if __name__ == '__main__':
    main()
