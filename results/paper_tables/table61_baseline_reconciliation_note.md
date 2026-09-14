# Baseline reconciliation note

Use **0.624** as the manuscript headline mean R2. This is the main BeamK30 ResidualMLP result from `table22_current_model_mean_r2_leaderboard.csv`.

The finite-bandpass section should report **delta values relative to its own internal raw rerun baseline**, not as a replacement for the manuscript headline. In that bandpass rerun, raw K30 mean R2 is **0.618** and Cw raw K30 is **0.665**. The Cw value is close to the manuscript Cw value of **0.661**, but the overall mean differs because the bandpass script is a separate diagnostic rerun.

Recommended caption language:

> Finite-bandpass variants are compared against the raw K=30 rerun baseline within the same simulation; therefore, the table emphasizes delta R2 rather than replacing the main manuscript performance table.

The best vegetation-index baseline has mean R2 **0.506**, below the main compact ResidualMLP mean R2 **0.624**.
