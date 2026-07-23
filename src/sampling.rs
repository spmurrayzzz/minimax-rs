use anyhow::{Result, bail};
use candle_core::{D, Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

pub const DEFAULT_TEMPERATURE: f64 = 1.0;
pub const DEFAULT_TOP_P: f64 = 0.95;
pub const DEFAULT_TOP_K: usize = 40;

/// Sampling settings after applying the built-in, server, and request-level
/// precedence rules.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            top_k: DEFAULT_TOP_K,
        }
    }
}

impl SamplingParams {
    pub fn validate(self) -> Result<Self> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            bail!("temperature must be a finite number greater than or equal to 0")
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            bail!("top_p must be a finite number between 0 and 1")
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: u32,
    logit: f32,
}

/// Per-request sampler. Its RNG is intentionally not shared with the model or
/// another request, so a request's number of generated tokens cannot perturb
/// later requests.
pub struct TokenSampler {
    params: SamplingParams,
    rng: StdRng,
}

/// Returns `ln p(token_id)` under the unfiltered model distribution.
///
/// Perplexity scoring deliberately ignores temperature, top-k, and top-p: it
/// measures the model's teacher-forced next-token distribution rather than a
/// request's sampling policy. The reduction and scalar selection stay on the
/// logits device, so tensor mode does not send the full vocabulary through the
/// controller process.
pub fn token_logprob(logits: &Tensor, token_id: u32) -> Result<f32> {
    let vocab_size = logits.dims1()?;
    if token_id as usize >= vocab_size {
        bail!("token {token_id} is outside logits vocabulary size {vocab_size}")
    }
    let value = candle_nn::ops::log_softmax(logits, D::Minus1)?
        .get(token_id as usize)?
        .to_scalar::<f32>()?;
    if !value.is_finite() {
        bail!("token {token_id} has non-finite log probability {value}")
    }
    Ok(value)
}

impl TokenSampler {
    pub fn new(params: SamplingParams, seed: u64) -> Result<Self> {
        Ok(Self {
            params: params.validate()?,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        // Keep deterministic decoding on-device. Besides avoiding unnecessary
        // sampling work, this preserves the fast path used by parity tests.
        if self.params.temperature == 0.0 {
            return Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?);
        }

        // llama.cpp applies top-k and top-p before temperature and distribution
        // sampling. Copying the full vocabulary is currently necessary only
        // for stochastic decoding; top-k reduces all later CPU work to a small
        // candidate set under the recommended defaults.
        let logits = logits.to_vec1::<f32>()?;
        if logits.is_empty() {
            bail!("cannot sample from empty logits")
        }
        if logits.iter().any(|logit| !logit.is_finite()) {
            bail!("cannot sample from non-finite logits")
        }

        let mut candidates = logits
            .into_iter()
            .enumerate()
            .map(|(id, logit)| Candidate {
                id: id as u32,
                logit,
            })
            .collect::<Vec<_>>();

        if self.params.top_k > 0 && self.params.top_k < candidates.len() {
            let top_k = self.params.top_k;
            candidates.select_nth_unstable_by(top_k, compare_candidates);
            candidates.truncate(top_k);
        }

        if self.params.top_p < 1.0 && candidates.len() > 1 {
            candidates.sort_unstable_by(compare_candidates);
            let weights = exponential_weights(&candidates, 1.0)?;
            let threshold = weights.iter().sum::<f64>() * self.params.top_p;
            let mut cumulative = 0.0;
            let mut keep = candidates.len();
            for (index, weight) in weights.into_iter().enumerate() {
                cumulative += weight;
                if cumulative >= threshold {
                    // Nucleus sampling includes the token that crosses top_p.
                    keep = index + 1;
                    break;
                }
            }
            candidates.truncate(keep.max(1));
        }

        let weights = exponential_weights(&candidates, self.params.temperature)?;
        let total = weights.iter().sum::<f64>();
        let mut sample = self.rng.random::<f64>() * total;
        let mut fallback = candidates[0].id;
        for (candidate, weight) in candidates.iter().zip(weights) {
            if weight > 0.0 {
                fallback = candidate.id;
            }
            if sample < weight {
                return Ok(candidate.id);
            }
            sample -= weight;
        }

        // Floating-point subtraction can leave a tiny positive remainder.
        Ok(fallback)
    }
}

fn compare_candidates(left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
    right
        .logit
        .total_cmp(&left.logit)
        .then_with(|| left.id.cmp(&right.id))
}

fn exponential_weights(candidates: &[Candidate], temperature: f64) -> Result<Vec<f64>> {
    let max_logit = candidates
        .iter()
        .map(|candidate| candidate.logit as f64)
        .max_by(f64::total_cmp)
        .ok_or_else(|| anyhow::anyhow!("cannot sample from an empty candidate set"))?;
    let weights = candidates
        .iter()
        .map(|candidate| (((candidate.logit as f64) - max_logit) / temperature).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        bail!("sampling probabilities are not finite")
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn logits(values: &[f32]) -> Tensor {
        Tensor::from_slice(values, values.len(), &Device::Cpu).expect("logits")
    }

    #[test]
    fn recommended_defaults_are_enabled() {
        assert_eq!(
            SamplingParams::default(),
            SamplingParams {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 40,
            }
        );
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let params = SamplingParams {
            temperature: 0.0,
            top_p: 0.01,
            top_k: 1,
        };
        let mut sampler = TokenSampler::new(params, 7).expect("sampler");
        assert_eq!(sampler.sample(&logits(&[0.0, 3.0, 2.0])).unwrap(), 1);
    }

    #[test]
    fn token_logprob_scores_the_unfiltered_distribution() {
        let values = logits(&[0.0, 3f32.ln()]);
        assert!((token_logprob(&values, 0).unwrap() - 0.25f32.ln()).abs() < 1e-6);
        assert!((token_logprob(&values, 1).unwrap() - 0.75f32.ln()).abs() < 1e-6);
        assert!(token_logprob(&values, 2).is_err());
    }

    #[test]
    fn top_k_one_is_greedy_at_positive_temperature() {
        let params = SamplingParams {
            temperature: 100.0,
            top_p: 1.0,
            top_k: 1,
        };
        for seed in 0..16 {
            let mut sampler = TokenSampler::new(params, seed).expect("sampler");
            assert_eq!(sampler.sample(&logits(&[0.0, 3.0, 2.0])).unwrap(), 1);
        }
    }

    #[test]
    fn top_p_excludes_tokens_after_the_crossing_candidate() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.7,
            top_k: 3,
        };
        // Within the top three, the unscaled probabilities are approximately
        // 0.665, 0.245, and 0.090. top_p therefore retains IDs 0 and 1.
        for seed in 0..64 {
            let mut sampler = TokenSampler::new(params, seed).expect("sampler");
            assert!(matches!(
                sampler.sample(&logits(&[3.0, 2.0, 1.0, 0.0])).unwrap(),
                0 | 1
            ));
        }
    }

    #[test]
    fn invalid_ranges_are_rejected() {
        for params in [
            SamplingParams {
                temperature: -0.1,
                ..SamplingParams::default()
            },
            SamplingParams {
                top_p: 1.1,
                ..SamplingParams::default()
            },
        ] {
            assert!(params.validate().is_err());
        }
    }
}
