//! Process-isolated tensor-parallel controller used by the HTTP server.
//!
//! The controller intentionally owns no CUDA state. It starts one long-lived
//! worker process per GPU, sends every cache-mutating command to both ranks,
//! and accepts an updated cache length only after both workers acknowledge it.

use crate::{
    sampling::{SamplingParams, TokenSampler},
    tensor_parallel::{TP_WORLD_SIZE, TensorParallelRankGroup},
    tensor_parallel_model::{
        HEAD_DIM, LOCAL_KV_HEADS, LOCAL_QUERY_HEADS, MAX_CONTEXT, TensorParallelRankModel,
        VOCAB_SIZE,
    },
};
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use cudarc::nccl::Id;
use serde::{Deserialize, Serialize};
use std::{
    ffi::c_char,
    io::{BufRead, BufReader, BufWriter, Write},
    os::fd::{AsRawFd, RawFd},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command as ProcessCommand, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

pub const WORKER_RANK_ARG: &str = "--internal-tp-rank";
pub const WORKER_NCCL_ID_ARG: &str = "--internal-tp-nccl-id";
pub const WORKER_DRY_RUN_ARG: &str = "--internal-tp-dry-run";

const WORLD_SIZE: usize = TP_WORLD_SIZE;
const WORKER_RESPONSE_TIMEOUT: Duration = Duration::from_secs(900);
const NCCL_SELF_TEST_ELEMENTS: usize = 8_193;

#[derive(Clone, Debug)]
pub struct TensorParallelInfo {
    pub world_size: usize,
    pub local_query_heads: usize,
    pub local_kv_heads: usize,
    pub head_dim: usize,
    pub nccl_version: i32,
    pub collective_backend: String,
    pub layer_weight_bytes: [usize; WORLD_SIZE],
    pub global_weight_bytes: [usize; WORLD_SIZE],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum WorkerCommand {
    SelfTest,
    Reset,
    Forward { ids: Vec<u32>, pos: usize },
    Truncate { seq_len: usize },
    BeginSampling { params: SamplingParams, seed: u64 },
    Sample,
    TokenLogprob { token_id: u32 },
    ClosePeerMapping,
    FreeLocalPeerBuffer,
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkerResponse {
    status: String,
    rank: Option<usize>,
    layer_weight_bytes: Option<usize>,
    global_weight_bytes: Option<usize>,
    nccl_version: Option<i32>,
    collective_backend: Option<String>,
    cache_len: Option<usize>,
    cache_shape: Option<Vec<usize>>,
    has_logits: Option<bool>,
    token_id: Option<u32>,
    token_logprob: Option<f32>,
    self_test_values: Option<Vec<f32>>,
    message: Option<String>,
}

impl WorkerResponse {
    fn ready(
        rank: usize,
        layer_weight_bytes: usize,
        global_weight_bytes: usize,
        nccl_version: i32,
        collective_backend: String,
    ) -> Self {
        Self {
            status: "ready".to_owned(),
            rank: Some(rank),
            layer_weight_bytes: Some(layer_weight_bytes),
            global_weight_bytes: Some(global_weight_bytes),
            nccl_version: Some(nccl_version),
            collective_backend: Some(collective_backend),
            cache_len: Some(0),
            cache_shape: None,
            has_logits: Some(false),
            token_id: None,
            token_logprob: None,
            self_test_values: None,
            message: None,
        }
    }

    fn done(cache_len: usize, cache_shape: Option<Vec<usize>>, has_logits: bool) -> Self {
        Self {
            status: "done".to_owned(),
            rank: None,
            layer_weight_bytes: None,
            global_weight_bytes: None,
            nccl_version: None,
            collective_backend: None,
            cache_len: Some(cache_len),
            cache_shape,
            has_logits: Some(has_logits),
            token_id: None,
            token_logprob: None,
            self_test_values: None,
            message: None,
        }
    }

    fn error(error: &anyhow::Error) -> Self {
        Self {
            status: "error".to_owned(),
            rank: None,
            layer_weight_bytes: None,
            global_weight_bytes: None,
            nccl_version: None,
            collective_backend: None,
            cache_len: None,
            cache_shape: None,
            has_logits: None,
            token_id: None,
            token_logprob: None,
            self_test_values: None,
            message: Some(format!("{error:#}")),
        }
    }

    fn ensure_ok(self, label: &str) -> Result<Self> {
        if self.status == "error" {
            bail!(
                "{label}: {}",
                self.message.as_deref().unwrap_or("worker failed")
            )
        }
        Ok(self)
    }
}

struct WorkerProcess {
    label: String,
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl WorkerProcess {
    fn spawn(model_dir: &Path, rank: usize, nccl_id: &str, dry_run: bool) -> Result<Self> {
        let executable = std::env::current_exe().context("resolve server executable")?;
        let mut command = ProcessCommand::new(executable);
        command
            .arg("--model")
            .arg(model_dir)
            .arg(WORKER_RANK_ARG)
            .arg(rank.to_string())
            .arg(WORKER_NCCL_ID_ARG)
            .arg(nccl_id)
            // Hidden workers do not use the public execution-mode setting.
            // Remove it so an overridden or stale parent environment cannot
            // make child argument parsing fail before worker dispatch.
            .env_remove("MINIMAX_PARALLELISM")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if dry_run {
            command.arg(WORKER_DRY_RUN_ARG);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;

            // Model loading can take minutes and deliberately happens before
            // Tokio installs graceful-shutdown handlers. Ensure a crash or
            // signal-killed controller cannot orphan GPU-owning rank workers.
            let parent_pid = unsafe { libc::getpid() };
            unsafe {
                command.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    // Keep terminal and process-group signals on the
                    // controller. Rank workers are shut down through their
                    // command pipes so CUDA IPC mappings close in order.
                    if libc::setpgid(0, 0) != 0
                        || libc::signal(libc::SIGINT, libc::SIG_IGN) == libc::SIG_ERR
                        || libc::signal(libc::SIGTERM, libc::SIG_IGN) == libc::SIG_ERR
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    // Close the fork/exec race where the parent exits between
                    // spawn and PR_SET_PDEATHSIG.
                    if libc::getppid() != parent_pid {
                        libc::_exit(1);
                    }
                    Ok(())
                });
            }
        }
        let label = format!("tensor rank {rank}");
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn {label} worker"))?;
        let input = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or_else(|| anyhow::anyhow!("{label} worker has no stdin"))?,
        );
        let output = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| anyhow::anyhow!("{label} worker has no stdout"))?,
        );
        Ok(Self {
            label,
            child,
            input,
            output,
        })
    }

    fn send(&mut self, command: &WorkerCommand) -> Result<()> {
        serde_json::to_writer(&mut self.input, command)
            .with_context(|| format!("serialize command for {}", self.label))?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;
        Ok(())
    }

    fn receive(&mut self) -> Result<WorkerResponse> {
        let mut line = String::new();
        if self.output.read_line(&mut line)? == 0 {
            let status = self.child.wait()?;
            bail!("{} worker exited unexpectedly with {status}", self.label)
        }
        serde_json::from_str::<WorkerResponse>(&line)
            .with_context(|| format!("decode {} response", self.label))?
            .ensure_ok(&self.label)
    }

    fn receive_timed(&mut self) -> Result<WorkerResponse> {
        let mut poll_fd = libc::pollfd {
            fd: self.output_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            let ready =
                unsafe { libc::poll(&mut poll_fd, 1, WORKER_RESPONSE_TIMEOUT.as_millis() as i32) };
            if ready < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error).context("poll tensor worker response");
            }
            if ready == 0 {
                self.terminate();
                bail!(
                    "timed out after {} seconds waiting for {}",
                    WORKER_RESPONSE_TIMEOUT.as_secs(),
                    self.label
                )
            }
            return self.receive();
        }
    }

    fn request(&mut self, command: &WorkerCommand) -> Result<WorkerResponse> {
        self.send(command)?;
        self.receive_timed()
    }

    fn output_fd(&self) -> RawFd {
        self.output.get_ref().as_raw_fd()
    }

    fn terminate(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// CUDA-free controller for the two process-owned tensor-parallel ranks.
pub struct TensorParallelController {
    ranks: Option<[WorkerProcess; WORLD_SIZE]>,
    info: TensorParallelInfo,
    ready: Arc<AtomicBool>,
    cache_len: usize,
    has_logits: bool,
}

impl TensorParallelController {
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::start(model_dir, false)
    }

    /// Initializes the production process topology, exercises both the IPC
    /// small-collective path and NCCL, then shuts both ranks down without
    /// loading any model tensor.
    pub fn dry_run(model_dir: &Path) -> Result<TensorParallelInfo> {
        let mut controller = Self::start(model_dir, true)?;
        let responses = controller.pair_request(&WorkerCommand::SelfTest)?;
        for (rank, response) in responses.iter().enumerate() {
            let values = response
                .self_test_values
                .as_deref()
                .with_context(|| format!("tensor rank {rank} returned no self-test values"))?;
            if values != [3.0, 3.0] {
                return controller.invalidate(anyhow::anyhow!(
                    "tensor rank {rank} collective self-test returned {values:?}, expected [3.0, 3.0]"
                ));
            }
        }
        let info = controller.info.clone();
        controller.shutdown()?;
        Ok(info)
    }

    fn start(model_dir: &Path, dry_run: bool) -> Result<Self> {
        let id = Id::new().context("create tensor-parallel NCCL unique ID")?;
        let encoded_id = encode_nccl_id(&id);
        let rank0 = WorkerProcess::spawn(model_dir, 0, &encoded_id, dry_run)?;
        let rank1 = match WorkerProcess::spawn(model_dir, 1, &encoded_id, dry_run) {
            Ok(worker) => worker,
            Err(error) => {
                drop(rank0);
                return Err(error);
            }
        };
        let mut ranks = [rank0, rank1];
        let ready = match receive_pair(&mut ranks) {
            Ok(ready) => ready,
            Err(error) => {
                terminate_pair(&mut ranks);
                return Err(error).context("initialize tensor-parallel rank workers");
            }
        };
        let info = match validate_ready(&ready) {
            Ok(info) => info,
            Err(error) => {
                terminate_pair(&mut ranks);
                return Err(error);
            }
        };
        Ok(Self {
            ranks: Some(ranks),
            info,
            ready: Arc::new(AtomicBool::new(true)),
            cache_len: 0,
            has_logits: false,
        })
    }

    pub fn info(&self) -> &TensorParallelInfo {
        &self.info
    }

    pub fn readiness_handle(&self) -> Arc<AtomicBool> {
        self.ready.clone()
    }

    pub fn reset(&mut self) -> Result<()> {
        let responses = self.pair_request(&WorkerCommand::Reset)?;
        self.accept_cache_responses(&responses, 0, false)?;
        self.cache_len = 0;
        self.has_logits = false;
        Ok(())
    }

    pub fn truncate_cache(&mut self, seq_len: usize) -> Result<()> {
        if seq_len > self.cache_len {
            bail!(
                "cannot truncate tensor cache from {} to {seq_len}",
                self.cache_len
            )
        }
        let responses = self.pair_request(&WorkerCommand::Truncate { seq_len })?;
        self.accept_cache_responses(&responses, seq_len, false)?;
        self.cache_len = seq_len;
        self.has_logits = false;
        Ok(())
    }

    pub fn forward(&mut self, ids: &[u32], pos: usize) -> Result<()> {
        validate_input_ids(ids, pos)?;
        if pos != self.cache_len {
            bail!(
                "tensor controller forward position {pos} does not match cache length {}",
                self.cache_len
            )
        }
        let expected_len = pos
            .checked_add(ids.len())
            .ok_or_else(|| anyhow::anyhow!("tensor cache length overflows usize"))?;
        let responses = self.pair_request(&WorkerCommand::Forward {
            ids: ids.to_vec(),
            pos,
        })?;
        self.accept_cache_responses(&responses, expected_len, true)?;
        self.cache_len = expected_len;
        self.has_logits = true;
        Ok(())
    }

    pub fn begin_sampling(&mut self, params: SamplingParams, seed: u64) -> Result<()> {
        if !self.has_logits {
            bail!("tensor rank 0 has no logits available for sampling")
        }
        let response = self.rank0_request(&WorkerCommand::BeginSampling { params, seed })?;
        self.accept_rank0_response(&response, true)?;
        Ok(())
    }

    pub fn sample(&mut self) -> Result<u32> {
        if !self.has_logits {
            bail!("tensor rank 0 has no logits available for sampling")
        }
        let response = self.rank0_request(&WorkerCommand::Sample)?;
        self.accept_rank0_response(&response, true)?;
        let token = response
            .token_id
            .context("tensor rank 0 returned no sampled token")?;
        if token >= VOCAB_SIZE as u32 {
            return self.invalidate(anyhow::anyhow!(
                "tensor rank 0 sampled token {token} outside vocabulary size {VOCAB_SIZE}"
            ));
        }
        Ok(token)
    }

    /// Scores one target while keeping the complete vocabulary in rank 0.
    pub fn token_logprob(&mut self, token_id: u32) -> Result<f32> {
        if !self.has_logits {
            bail!("tensor rank 0 has no logits available for scoring")
        }
        if token_id >= VOCAB_SIZE as u32 {
            bail!("score token {token_id} is outside vocabulary size {VOCAB_SIZE}")
        }
        let response = self.rank0_request(&WorkerCommand::TokenLogprob { token_id })?;
        self.accept_rank0_response(&response, true)?;
        let value = response
            .token_logprob
            .context("tensor rank 0 returned no token log probability")?;
        if !value.is_finite() {
            return self.invalidate(anyhow::anyhow!(
                "tensor rank 0 returned non-finite token log probability {value}"
            ));
        }
        Ok(value)
    }

    fn accept_cache_responses(
        &mut self,
        responses: &[WorkerResponse; WORLD_SIZE],
        expected_len: usize,
        rank0_has_logits: bool,
    ) -> Result<()> {
        let expected_shape = [1, LOCAL_KV_HEADS, expected_len, HEAD_DIM];
        let shapes_match = responses[0].cache_shape == responses[1].cache_shape;
        let shape_is_valid = responses[0]
            .cache_shape
            .as_deref()
            .map_or(expected_len == 0, |shape| shape == expected_shape);
        let valid = responses[0].cache_len == Some(expected_len)
            && responses[1].cache_len == Some(expected_len)
            && responses[0].has_logits == Some(rank0_has_logits)
            && responses[1].has_logits == Some(false)
            && shapes_match
            && shape_is_valid;
        if !valid {
            return self.invalidate(anyhow::anyhow!(
                "tensor ranks returned inconsistent cache state: expected len {expected_len}, shape {expected_shape:?}, and logits [{rank0_has_logits}, false]; got len [{:?}, {:?}], shape [{:?}, {:?}], and logits [{:?}, {:?}]",
                responses[0].cache_len,
                responses[1].cache_len,
                responses[0].cache_shape,
                responses[1].cache_shape,
                responses[0].has_logits,
                responses[1].has_logits,
            ));
        }
        Ok(())
    }

    fn accept_rank0_response(&mut self, response: &WorkerResponse, has_logits: bool) -> Result<()> {
        if response.cache_len != Some(self.cache_len) || response.has_logits != Some(has_logits) {
            return self.invalidate(anyhow::anyhow!(
                "tensor rank 0 returned inconsistent sampling state: expected cache {} and has_logits={has_logits}, got cache {:?} and has_logits={:?}",
                self.cache_len,
                response.cache_len,
                response.has_logits,
            ));
        }
        Ok(())
    }

    fn pair_request(&mut self, command: &WorkerCommand) -> Result<[WorkerResponse; WORLD_SIZE]> {
        let result = match self.ranks.as_mut() {
            Some(ranks) => pair_request(ranks, command),
            None => bail!("tensor-parallel rank workers are unavailable"),
        };
        match result {
            Ok(responses) => Ok(responses),
            Err(error) => self.invalidate(error),
        }
    }

    fn rank0_request(&mut self, command: &WorkerCommand) -> Result<WorkerResponse> {
        let result = match self.ranks.as_mut() {
            Some(ranks) => ranks[0].request(command),
            None => bail!("tensor-parallel rank workers are unavailable"),
        };
        match result {
            Ok(response) => Ok(response),
            Err(error) => self.invalidate(error),
        }
    }

    fn invalidate<T>(&mut self, error: anyhow::Error) -> Result<T> {
        self.ready.store(false, Ordering::Release);
        if let Some(mut ranks) = self.ranks.take() {
            terminate_pair(&mut ranks);
        }
        self.has_logits = false;
        Err(error)
    }

    fn shutdown(&mut self) -> Result<()> {
        self.ready.store(false, Ordering::Release);
        let Some(mut ranks) = self.ranks.take() else {
            return Ok(());
        };
        match shutdown_rank_pair(&mut ranks) {
            Ok(()) => Ok(()),
            Err(error) => {
                terminate_pair(&mut ranks);
                Err(error)
            }
        }
    }
}

impl Drop for TensorParallelController {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn validate_ready(responses: &[WorkerResponse; WORLD_SIZE]) -> Result<TensorParallelInfo> {
    for (rank, response) in responses.iter().enumerate() {
        if response.status != "ready" || response.rank != Some(rank) {
            bail!(
                "invalid tensor worker ready response for rank {rank}: status={} rank={:?}",
                response.status,
                response.rank
            )
        }
        if response.cache_len != Some(0) || response.has_logits != Some(false) {
            bail!("tensor rank {rank} did not start with an empty cache")
        }
    }
    let nccl_version = responses[0]
        .nccl_version
        .context("tensor rank 0 omitted NCCL version")?;
    if responses[1].nccl_version != Some(nccl_version) {
        bail!(
            "tensor ranks reported different NCCL versions: {:?} and {:?}",
            responses[0].nccl_version,
            responses[1].nccl_version
        )
    }
    let collective_backend = responses[0]
        .collective_backend
        .clone()
        .context("tensor rank 0 omitted collective backend")?;
    if responses[1].collective_backend.as_deref() != Some(collective_backend.as_str()) {
        bail!(
            "tensor ranks reported different collective backends: {:?} and {:?}",
            responses[0].collective_backend,
            responses[1].collective_backend
        )
    }
    Ok(TensorParallelInfo {
        world_size: WORLD_SIZE,
        local_query_heads: LOCAL_QUERY_HEADS,
        local_kv_heads: LOCAL_KV_HEADS,
        head_dim: HEAD_DIM,
        nccl_version,
        collective_backend,
        layer_weight_bytes: [
            responses[0].layer_weight_bytes.unwrap_or_default(),
            responses[1].layer_weight_bytes.unwrap_or_default(),
        ],
        global_weight_bytes: [
            responses[0].global_weight_bytes.unwrap_or_default(),
            responses[1].global_weight_bytes.unwrap_or_default(),
        ],
    })
}

fn validate_input_ids(ids: &[u32], pos: usize) -> Result<()> {
    if ids.is_empty() {
        bail!("tensor model input must contain at least one token")
    }
    if let Some((index, id)) = ids
        .iter()
        .enumerate()
        .find(|(_, id)| **id >= VOCAB_SIZE as u32)
    {
        bail!(
            "tensor model input token {id} at position {index} is outside vocabulary size {VOCAB_SIZE}"
        )
    }
    let end = pos
        .checked_add(ids.len())
        .ok_or_else(|| anyhow::anyhow!("tensor model input position overflows usize"))?;
    if end > MAX_CONTEXT {
        bail!("tensor model input ends at {end}, beyond context limit {MAX_CONTEXT}")
    }
    Ok(())
}

/// Entry point for a hidden rank-worker invocation of the server executable.
pub fn run_worker(model_dir: &Path, rank: usize, encoded_id: &str, dry_run: bool) -> Result<()> {
    if rank >= WORLD_SIZE {
        bail!("tensor worker rank must be 0 or 1, got {rank}")
    }
    let id = decode_nccl_id(encoded_id)?;
    let result = if dry_run {
        run_dry_worker(rank, id)
    } else {
        run_model_worker(model_dir, rank, id)
    };
    if let Err(error) = &result {
        // Initialization errors happen before worker_loop can report them.
        // A duplicate error after a command failure is harmless because the
        // controller consumes the first response and terminates both ranks.
        let _ = write_response(&WorkerResponse::error(error));
    }
    result
}

fn run_model_worker(model_dir: &Path, rank: usize, id: Id) -> Result<()> {
    let shards = crate::model_files::discover_gguf_shards(model_dir)?;
    let device = Device::new_cuda(rank)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let mut model = TensorParallelRankModel::load(&shards, rank, &device, id)?;
    device.synchronize()?;
    write_response(&WorkerResponse::ready(
        rank,
        model.layer_weight_bytes(),
        model.global_weight_bytes(),
        model.nccl_version(),
        model.collective_backend().to_owned(),
    ))?;

    let mut latest_logits = None;
    let mut sampler = None;
    worker_loop(|command| match command {
        WorkerCommand::Reset => {
            model.reset();
            latest_logits = None;
            sampler = None;
            Ok(WorkerResponse::done(0, None, false))
        }
        WorkerCommand::Forward { ids, pos } => {
            let output = model.forward(&ids, pos)?;
            device.synchronize()?;
            let cache_len = model.cache_length()?;
            latest_logits = output.logits;
            let has_logits = latest_logits.is_some();
            Ok(WorkerResponse::done(
                cache_len,
                model.cache_shape()?,
                has_logits,
            ))
        }
        WorkerCommand::Truncate { seq_len } => {
            model.truncate_cache(seq_len)?;
            latest_logits = None;
            sampler = None;
            Ok(WorkerResponse::done(seq_len, model.cache_shape()?, false))
        }
        WorkerCommand::BeginSampling { params, seed } => {
            if rank != 0 {
                bail!("sampling initialization is rank-0-only")
            }
            if latest_logits.is_none() {
                bail!("rank 0 has no logits available for sampling")
            }
            sampler = Some(TokenSampler::new(params, seed)?);
            Ok(WorkerResponse::done(model.cache_length()?, None, true))
        }
        WorkerCommand::Sample => {
            if rank != 0 {
                bail!("sampling is rank-0-only")
            }
            let token_id = sampler
                .as_mut()
                .context("rank 0 sampler has not been initialized")?
                .sample(latest_logits.as_ref().context("rank 0 has no logits")?)?;
            let mut response = WorkerResponse::done(model.cache_length()?, None, true);
            response.token_id = Some(token_id);
            Ok(response)
        }
        WorkerCommand::TokenLogprob { token_id } => {
            if rank != 0 {
                bail!("token scoring is rank-0-only")
            }
            let value = crate::sampling::token_logprob(
                latest_logits.as_ref().context("rank 0 has no logits")?,
                token_id,
            )?;
            let mut response = WorkerResponse::done(model.cache_length()?, None, true);
            response.token_logprob = Some(value);
            Ok(response)
        }
        WorkerCommand::ClosePeerMapping => {
            model.close_peer_mapping()?;
            Ok(WorkerResponse::done(
                model.cache_length()?,
                model.cache_shape()?,
                latest_logits.is_some(),
            ))
        }
        WorkerCommand::FreeLocalPeerBuffer => {
            model.free_local_peer_buffer()?;
            Ok(WorkerResponse::done(
                model.cache_length()?,
                model.cache_shape()?,
                latest_logits.is_some(),
            ))
        }
        WorkerCommand::SelfTest => bail!("loaded tensor worker received dry-run self-test"),
        WorkerCommand::Shutdown => unreachable!("shutdown is handled by worker_loop"),
    })
}

fn run_dry_worker(rank: usize, id: Id) -> Result<()> {
    let device = Device::new_cuda(rank)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let mut collective = TensorParallelRankGroup::new(&device, rank, id)?;
    device.synchronize()?;
    write_response(&WorkerResponse::ready(
        rank,
        0,
        0,
        collective.nccl_version(),
        collective.backend().to_owned(),
    ))?;

    worker_loop(|command| match command {
        WorkerCommand::SelfTest => {
            let mut observed = Vec::with_capacity(2);
            for count in [2, NCCL_SELF_TEST_ELEMENTS] {
                let input = vec![rank as f32 + 1.0; count];
                let input = Tensor::from_vec(input, count, &device)?;
                let output = collective.all_reduce_sum(input)?;
                device.synchronize()?;
                let values = output.to_vec1::<f32>()?;
                if values.iter().any(|&value| value != 3.0) {
                    bail!("rank {rank} collective self-test failed for {count} elements")
                }
                observed.push(values[0]);
            }
            let mut response = WorkerResponse::done(0, None, false);
            response.self_test_values = Some(observed);
            Ok(response)
        }
        WorkerCommand::ClosePeerMapping => {
            collective.close_peer_mapping()?;
            Ok(WorkerResponse::done(0, None, false))
        }
        WorkerCommand::FreeLocalPeerBuffer => {
            collective.free_local_peer_buffer()?;
            Ok(WorkerResponse::done(0, None, false))
        }
        WorkerCommand::Reset
        | WorkerCommand::Forward { .. }
        | WorkerCommand::Truncate { .. }
        | WorkerCommand::BeginSampling { .. }
        | WorkerCommand::Sample
        | WorkerCommand::TokenLogprob { .. } => {
            bail!("dry-run tensor worker received model command")
        }
        WorkerCommand::Shutdown => unreachable!("shutdown is handled by worker_loop"),
    })
}

fn worker_loop(mut handle: impl FnMut(WorkerCommand) -> Result<WorkerResponse>) -> Result<()> {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let command: WorkerCommand =
            serde_json::from_str(&line?).context("decode tensor worker command")?;
        if matches!(command, WorkerCommand::Shutdown) {
            let mut response = WorkerResponse::done(0, None, false);
            response.status = "shutdown".to_owned();
            write_response(&response)?;
            return Ok(());
        }
        match handle(command) {
            Ok(response) => write_response(&response)?,
            Err(error) => {
                write_response(&WorkerResponse::error(&error))?;
                return Err(error);
            }
        }
    }
    bail!("tensor worker command pipe closed without shutdown")
}

fn write_response(response: &WorkerResponse) -> Result<()> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, response)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn pair_request(
    ranks: &mut [WorkerProcess; WORLD_SIZE],
    command: &WorkerCommand,
) -> Result<[WorkerResponse; WORLD_SIZE]> {
    if let Err(error) = ranks[0].send(command).and_then(|()| ranks[1].send(command)) {
        terminate_pair(ranks);
        return Err(error).context("send paired tensor-worker command");
    }
    receive_pair(ranks)
}

fn receive_pair(ranks: &mut [WorkerProcess; WORLD_SIZE]) -> Result<[WorkerResponse; WORLD_SIZE]> {
    let deadline = Instant::now() + WORKER_RESPONSE_TIMEOUT;
    let mut responses = [None, None];
    while responses.iter().any(Option::is_none) {
        let now = Instant::now();
        if now >= deadline {
            terminate_pair(ranks);
            bail!("timed out waiting for tensor rank workers")
        }
        let timeout_ms = deadline
            .saturating_duration_since(now)
            .as_millis()
            .min(i32::MAX as u128) as i32;
        let mut poll_fds = [
            libc::pollfd {
                fd: if responses[0].is_none() {
                    ranks[0].output_fd()
                } else {
                    -1
                },
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: if responses[1].is_none() {
                    ranks[1].output_fd()
                } else {
                    -1
                },
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        let ready = unsafe { libc::poll(poll_fds.as_mut_ptr(), WORLD_SIZE as _, timeout_ms) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            terminate_pair(ranks);
            return Err(error).context("poll tensor rank worker responses");
        }
        if ready == 0 {
            terminate_pair(ranks);
            bail!("timed out waiting for tensor rank workers")
        }
        for rank in 0..WORLD_SIZE {
            let terminal = libc::POLLIN | libc::POLLHUP | libc::POLLERR | libc::POLLNVAL;
            if responses[rank].is_none() && poll_fds[rank].revents & terminal != 0 {
                match ranks[rank].receive() {
                    Ok(response) => responses[rank] = Some(response),
                    Err(error) => {
                        terminate_pair(ranks);
                        return Err(error)
                            .with_context(|| format!("receive tensor rank {rank} response"));
                    }
                }
            }
        }
    }
    let [Some(rank0), Some(rank1)] = responses else {
        unreachable!("both tensor rank responses were populated")
    };
    Ok([rank0, rank1])
}

fn shutdown_rank_pair(ranks: &mut [WorkerProcess; WORLD_SIZE]) -> Result<()> {
    // Every CUDA IPC importer must close its mapping before either exporter
    // frees the underlying allocation.
    pair_request(ranks, &WorkerCommand::ClosePeerMapping)?;
    pair_request(ranks, &WorkerCommand::FreeLocalPeerBuffer)?;
    let shutdown = pair_request(ranks, &WorkerCommand::Shutdown)?;
    if shutdown[0].status != "shutdown" || shutdown[1].status != "shutdown" {
        bail!("tensor rank workers returned invalid shutdown acknowledgements")
    }
    let status0 = ranks[0].child.wait()?;
    let status1 = ranks[1].child.wait()?;
    if !status0.success() || !status1.success() {
        bail!("tensor rank workers exited with {status0} and {status1}")
    }
    Ok(())
}

fn terminate_pair(ranks: &mut [WorkerProcess; WORLD_SIZE]) {
    for rank in ranks.iter_mut() {
        rank.terminate();
    }
    for rank in ranks.iter_mut() {
        let _ = rank.child.wait();
    }
}

fn encode_nccl_id(id: &Id) -> String {
    let mut encoded = String::with_capacity(256);
    for &byte in id.internal() {
        use std::fmt::Write as _;
        write!(&mut encoded, "{:02x}", byte as u8).expect("write NCCL ID hex");
    }
    encoded
}

fn decode_nccl_id(encoded: &str) -> Result<Id> {
    if encoded.len() != 256 {
        bail!("NCCL ID must contain 256 hexadecimal characters")
    }
    let mut internal = [0 as c_char; 128];
    for (index, byte) in internal.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16)
            .with_context(|| format!("decode NCCL ID byte {index}"))? as c_char;
    }
    Ok(Id::uninit(internal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_input_validation_rejects_invalid_sequences_without_workers() {
        assert!(validate_input_ids(&[0, VOCAB_SIZE as u32 - 1], 0).is_ok());
        assert!(validate_input_ids(&[], 0).is_err());
        assert!(validate_input_ids(&[VOCAB_SIZE as u32], 0).is_err());
        assert!(validate_input_ids(&[0, 1], MAX_CONTEXT - 1).is_err());
    }

    #[test]
    fn nccl_id_decoder_rejects_malformed_text() {
        assert!(decode_nccl_id("00").is_err());
        assert!(decode_nccl_id(&"z".repeat(256)).is_err());
    }

    #[test]
    fn fatal_invalidation_clears_shared_readiness() {
        let ready = Arc::new(AtomicBool::new(true));
        let mut controller = TensorParallelController {
            ranks: None,
            info: TensorParallelInfo {
                world_size: WORLD_SIZE,
                local_query_heads: LOCAL_QUERY_HEADS,
                local_kv_heads: LOCAL_KV_HEADS,
                head_dim: HEAD_DIM,
                nccl_version: 0,
                collective_backend: "test".to_owned(),
                layer_weight_bytes: [0; WORLD_SIZE],
                global_weight_bytes: [0; WORLD_SIZE],
            },
            ready: ready.clone(),
            cache_len: 0,
            has_logits: false,
        };

        let error = controller
            .invalidate::<()>(anyhow::anyhow!("rank failed"))
            .expect_err("invalidation must return the rank error");
        assert_eq!(error.to_string(), "rank failed");
        assert!(!ready.load(Ordering::Acquire));
    }
}
