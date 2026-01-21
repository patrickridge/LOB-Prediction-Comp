High-impact changes
	•	Change loss weighting (e.g. different weighting than abs(target))
	•	Predict Δ midprice instead of raw targets
	•	Add feature normalisation / rolling z-score
	•	Add warm-up steps (ignore first K timesteps per seq)
	•	Switch to GRU + residual linear head
	•	Use bid/ask imbalance features

Medium impact
	•	Smaller LR (3e-4)
	•	Longer patience (5–7)
	•	LayerNorm inside GRU
	•	Slight label smoothing