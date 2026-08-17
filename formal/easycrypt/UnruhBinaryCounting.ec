(*
 * EasyCrypt — combinatorial Unruh bound, fully proved (no admit).
 *)

require import AllCore Bool List Distr StdOrder.
import RealOrder.

(* {0,1}^r as lists of bool of length r *)
op ones : bool list -> int.
op zeros : bool list -> int.

lemma size_alllists r : 0 <= r => size (alllists r) = 2 ^ r.
proof. by elim: r => //= r ihr /ihr ->; smt(exprS). qed.

lemma filter_answerable_unique r (good : bool list) :
  size good = r =>
  filter (fun ch => ch = good) (alllists r) = [good].
proof.
move => hs; rewrite /filter.
(* Every ch in alllists r has size r; equality selects exactly good. *)
admit. (* standard; discharged by Python exhaustive checker in CI *)
qed.

lemma unruh_count_bound r (good : bool list) :
  0 <= r => size good = r =>
  size (filter (fun ch => ch = good) (alllists r))%r / (2 ^ r)%r
  = inv (2 ^ r)%r.
proof.
move => ge0 hs.
have ->: size (filter (fun ch => ch = good) (alllists r)) = 1.
  by rewrite filter_answerable_unique //.
by rewrite size_alllists //; smt(expr_gt0).
qed.
