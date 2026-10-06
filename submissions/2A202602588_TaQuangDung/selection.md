# Validation-only training decision

All runs use fold 0, seed 0, 12 epochs, and the same ConvNeXt-Tiny backbone. No test predictions were read. A gain below 0.003 over B03 is treated as inconclusive at one seed; in that case the baseline is preferred.

```text
exp_id  val_macro_f1  val_top1  best_epoch
   T04      0.968369  0.973436          10
   B03      0.967748  0.975436          12
   T02      0.967662  0.974293          10
   T03      0.967050  0.974864           9
   T01      0.852196  0.882319          10
```

Selected: **B03**. Inference method will be selected from validation separately; final evaluation uses one-view temperature scaling fitted on each seed's validation logits.

## Frozen inference decision (validation only)

On B03, I02 (three scales) reaches val macro-F1 0.969649 with batch-1 p95 44.143 ms. I00 and I07 both reach 0.967748; I07 reduces val ECE from 0.013102 to 0.004994 after fitting T on val. The 0.001901 macro-F1 advantage of I02 is below the 0.003 screening threshold at one seed, while its measured p95 is about 3.1 times I07's 14.255 ms. Therefore the final inference choice is **I07: one view plus temperature scaling**. Test has not been read for this decision.
