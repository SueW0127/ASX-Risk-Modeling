# ASX-Risk-Modeling

Time series risk modelling on the ASX200 using volatility, drawdowns, and regime aware ML with walk forward validation.

This simple project builds a reproducible risk modelling pipeline for the ASX 200. Using historical index data, the model estimates short-horizon downside risk based on volatility, drawdown and momentum features with walk forward validation. Whilst the project focuses on the ASX 200, it can be adjusted to apply to any stock given sufficient historical index data.

**Problem statement**
Given historical ASX price data up to time t, can we estimate whether market risk over the next H trading days is elevated (large drawdown or volatility spike) using only information available at time t?

**Why this matters**
Downside risk estimation is critical for portfolio allocation, stress testing, and risk adjusted decision making. Unlike return prediction, risk modelling focuses on uncertainty and tail behaviour, which is more aligned with real world risk management.
