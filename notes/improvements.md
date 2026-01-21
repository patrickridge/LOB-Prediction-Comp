# High-impact changes
	•	Change loss weighting (e.g. different weighting than abs(target))
	•	Predict Δ midprice instead of raw targets
	•	Add feature normalisation / rolling z-score
	•	Add warm-up steps (ignore first K timesteps per seq)
	•	Switch to GRU + residual linear head
	•	Use bid/ask imbalance features

# Medium impact
	•	Smaller LR (3e-4)
	•	Longer patience (5–7)
	•	LayerNorm inside GRU
	•	Slight label smoothing

## Best improvements to try next
	•	Match the loss to the metric
Train with a correlation-style loss (weighted Pearson) or a hybrid loss instead of pure weighted MSE.
	•	Train only where it’s scored
Use windows ending at need_prediction==1, or heavily up-weight those steps so the model focuses on what matters.
	•	Handle regime changes explicitly
Add a simple regime signal (e.g. recent volatility/volume stats) or use a small mixture-of-experts output head.
	•	Reduce over-regularisation in deep GRUs
Lower dropout (or remove it) and rely on early stopping + gradient clipping instead.
	•	Change capacity, not just depth
Try fewer layers with larger hidden size, or add LayerNorm around GRU outputs.
	•	Improve optimisation
Use AdamW + LR schedule (cosine / OneCycle) instead of fixed LR.

If you do only two things:
	1.	correlation-based loss
	2.	windowed training on prediction points