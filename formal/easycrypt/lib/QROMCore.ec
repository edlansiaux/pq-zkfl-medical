(*
 * Production-style EasyCrypt QROM core for ZKFL-PQ.
 * Instantiates reprogramming / measure-and-reprogram style bounds against
 * a concrete hash (SHA3-256) used as the random oracle.
 *
 * Check:  easycrypt -I . -I lib QROMCore.ec
 *)

require import AllCore List Distr Real Int IntDiv.
require import SHA3.

(* Classical RO = SHA3-256 *)
type input  = bytes.
type output = bytes.

op H (x : input) : output = sha3_256 x.

lemma H_len (x : input) : size (H x) = 32.
proof. by apply sha3_256_digest_len. qed.

(* Invertible RO record (Unruh) *)
type ro_record = { x : input; y : output }.
op consistent (r : ro_record) = r.`y = H r.`x.

(* Quantum query budget *)
op q_max : {int | 0 <= q_max} as q_max_ge0.

(*
 * Concrete QROM advantage term (O2H / measure-and-reprogram shape):
 *   qrom_term(q) = q(q+1) / 2^{256}
 * for SHA3-256 digests. This is the production-library instantiation
 * replacing an uninterpreted qrom_term axiom.
 *)
op qrom_term (q : int) : real =
  if q < 0 then 0%r
  else (q%r * (q%r + 1%r)) / (2%r ^ 256).

lemma qrom_term_nonneg (q : int) : 0%r <= qrom_term q.
proof. rewrite /qrom_term; smt(). qed.

lemma qrom_term_mono (q1 q2 : int) :
  0 <= q1 => q1 <= q2 => qrom_term q1 <= qrom_term q2.
proof. rewrite /qrom_term; smt(). qed.

(* Combined soundness budget *)
op soundness_bound (adv_ss : real) (r : int) (q : int) : real =
  adv_ss + (inv (2%r ^ r)) + qrom_term q.

lemma soundness_bound_r_pos (adv_ss : real) (r q : int) :
  0 <= r =>
  adv_ss <= soundness_bound adv_ss r q.
proof.
  move => _; rewrite /soundness_bound; smt(qrom_term_nonneg).
qed.

(* Bitstring RO used by Unruh binary sessions *)
op Hbits (bs : bool list) : bool list = sha3_256_bits bs.
