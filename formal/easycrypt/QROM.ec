(*
 * Minimal in-repo QROM interface for ZKFL-PQ Unruh theories.
 * Self-contained: no external EasyCrypt QROM stdlib required.
 *
 * Models Unruh-style invertible random oracles and a reprogramming bound
 * used by UnruhBinaryQROM.ec.
 *)

require import AllCore List Distr Real.

type input.
type output.

(* Classical RO view *)
op H : input -> output.

(* Invertible RO record (Unruh): adversary may see (x, H(x)) pairs *)
type ro_record = { x : input; y : output }.

op consistent (r : ro_record) = r.`y = H r.`x.

(* Abstract QROM query budget *)
op q_max : {int | 0 <= q_max} as q_max_ge0.

(* Reprogramming / measure-and-reprogram advantage term (axiomatized shape) *)
op qrom_term : int -> real.
axiom qrom_term_nonneg (q : int) : 0%r <= qrom_term q.
axiom qrom_term_mono (q1 q2 : int) : q1 <= q2 => qrom_term q1 <= qrom_term q2.

(* Soundness budget combining special-soundness, Unruh counting, QROM *)
op soundness_bound (adv_ss : real) (r : int) (q : int) : real =
  adv_ss + (inv (2%r ^ r)) + qrom_term q.

lemma soundness_bound_r_pos (adv_ss : real) (r q : int) :
  0 <= r =>
  adv_ss <= soundness_bound adv_ss r q.
proof.
  move => _; rewrite /soundness_bound; smt(qrom_term_nonneg).
qed.
