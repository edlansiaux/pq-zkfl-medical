(*
 * SHA3 instantiation of the QROM library (production link).
 * Unruh theories import this module (via QROM.ec) so H is SHA3-256.
 *
 * Check:  easycrypt -I . -I lib SHA3_QROM.ec
 *)

require import AllCore.
require import QROMCore.
require import SHA3.

(* Re-export: H is already sha3_256 in QROMCore *)
lemma sha3_is_ro (m : bytes) : H m = sha3_256 m.
proof. by rewrite /H. qed.
