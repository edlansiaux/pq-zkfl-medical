(*
 * QROM interface for ZKFL-PQ Unruh theories.
 * Now links the production SHA3-instantiated QROM library (lib/QROMCore.ec
 * + lib/SHA3.ec) rather than an uninterpreted RO.
 *
 * Check:  easycrypt -I . -I lib QROM.ec
 *)

require import AllCore List Distr Real.
require import QROMCore.
require import SHA3.

(* Types / ops re-exported for UnruhBinaryQROM.ec *)
(* H, ro_record, consistent, q_max, qrom_term, soundness_bound, Hbits *)
