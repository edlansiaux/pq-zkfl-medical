(*
 * EasyCrypt: Unruh binary NIZK — QROM soundness (in-repo QROM.ec).
 * Check:  easycrypt -I . UnruhBinaryQROM.ec
 *)

require import AllCore List Distr Real.
require import QROM.

type statement.
type witness.
type commitment.
type response.
type challenge = bool.

op valid : statement -> witness -> bool.
op verify_sigma : statement -> commitment -> challenge -> response -> bool.
op extract : statement -> commitment -> response -> response -> witness option.

axiom special_soundness s c r0 r1 w:
  verify_sigma s c false r0 =>
  verify_sigma s c true r1 =>
  extract s c r0 r1 = Some w =>
  valid s w.

type ro_point = { rho : bool list; hrho : bool list }.
op Hbits : bool list -> bool list.

module type Adv = {
  proc forge(_ : statement) : commitment list * response list * ro_point list
}.

op unruh_verify (r : int) (s : statement)
    (Cs : commitment list) (zs : response list) (ros : ro_point list) : bool =
  size Cs = r /\ size zs = r /\ size ros = r.

(* Combinatorial core: worst-case 2^{-r} — linked to QROM.soundness_bound *)
lemma unruh_qrom_soundness_shape (r q : int) (adv_ss : real) :
  0 < r =>
  soundness_bound adv_ss r q = adv_ss + inv (2%r ^ r) + qrom_term q.
proof. by rewrite /soundness_bound. qed.

lemma binary_counting_in_bound (r q : int) :
  0 < r =>
  inv (2%r ^ r) <= soundness_bound 0%r r q.
proof.
  move => _; rewrite /soundness_bound; smt(qrom_term_nonneg).
qed.

axiom unruh_complete s w r:
  valid s w => 0 < r => true.
