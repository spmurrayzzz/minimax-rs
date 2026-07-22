use anyhow::{Context, Result, bail};
use candle_core::{D, Device, Tensor};
use clap::{Parser, ValueEnum};
use cudarc::nccl::Id;
use minimax::{
    chat::{self, ChatMessage},
    model::Model as PipelineModel,
    model_files::discover_gguf_shards,
    tensor_parallel_model::{HEAD_DIM, LOCAL_KV_HEADS, TensorParallelRankModel},
    tokenizer::MiniMaxTokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    ffi::c_char,
    io::{BufRead, BufReader, BufWriter, Write},
    os::fd::AsRawFd,
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command as ProcessCommand, Stdio},
    time::{Duration, Instant},
};

const WORKER_RESPONSE_TIMEOUT: Duration = Duration::from_secs(900);
const GOLDEN_TEST_TOKENS: [u32; 24] = [
    758, 3100, 3700, 494, 4500, 4969, 1204, 355, 258, 4160, 1618, 4838, 46, 517, 23413, 1352, 7623,
    36238, 46, 25209, 687, 5177, 23077, 46,
];

#[derive(Clone, Copy, Debug, ValueEnum)]
enum WorkerKind {
    Pipeline,
    Rank,
}

#[derive(Debug, Parser)]
struct Args {
    /// Directory containing the four split GGUF files.
    #[arg(long, env = "MINIMAX_MODEL_DIR", value_name = "DIR")]
    model: PathBuf,
    /// Chat message used for end-to-end greedy validation.
    #[arg(long, default_value = "test")]
    message: String,
    #[arg(long, default_value_t = 24)]
    max_tokens: usize,
    #[arg(long, default_value_t = 512)]
    prefill_chunk: usize,
    /// Skip the existing 31/31 pipeline baseline to save one model load.
    #[arg(long)]
    skip_pipeline: bool,
    /// Also reset and validate fresh 1/5/39/512/513-token forwards.
    #[arg(long)]
    validate_prefill_lengths: bool,
    /// Prefill through 8,191 and decode across the 8,192 attention split transition.
    #[arg(long)]
    check_transition: bool,
    #[arg(long, value_enum, hide = true)]
    worker: Option<WorkerKind>,
    #[arg(long, hide = true)]
    rank: Option<usize>,
    #[arg(long, hide = true)]
    nccl_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum WorkerCommand {
    Reset,
    Forward {
        ids: Vec<u32>,
        pos: usize,
        return_logits: bool,
        return_hidden: bool,
        return_memory: bool,
    },
    Truncate {
        seq_len: usize,
    },
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
    token_id: Option<u32>,
    logits: Option<Vec<f32>>,
    hidden: Option<Vec<f32>>,
    milliseconds: Option<f64>,
    memory_used: Option<Vec<usize>>,
    message: Option<String>,
}

impl WorkerResponse {
    fn ready(
        rank: Option<usize>,
        layer_weight_bytes: Option<usize>,
        global_weight_bytes: Option<usize>,
        nccl_version: Option<i32>,
        collective_backend: Option<String>,
        memory_used: Vec<usize>,
    ) -> Self {
        Self {
            status: "ready".to_owned(),
            rank,
            layer_weight_bytes,
            global_weight_bytes,
            nccl_version,
            collective_backend,
            cache_len: Some(0),
            cache_shape: None,
            token_id: None,
            logits: None,
            hidden: None,
            milliseconds: None,
            memory_used: Some(memory_used),
            message: None,
        }
    }

    fn done(cache_len: usize, cache_shape: Option<Vec<usize>>) -> Self {
        Self {
            status: "done".to_owned(),
            rank: None,
            layer_weight_bytes: None,
            global_weight_bytes: None,
            nccl_version: None,
            collective_backend: None,
            cache_len: Some(cache_len),
            cache_shape,
            token_id: None,
            logits: None,
            hidden: None,
            milliseconds: None,
            memory_used: None,
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
            token_id: None,
            logits: None,
            hidden: None,
            milliseconds: None,
            memory_used: None,
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
    fn spawn(args: &Args, kind: WorkerKind, rank: Option<usize>, id: Option<&str>) -> Result<Self> {
        let executable = std::env::current_exe().context("resolve model benchmark executable")?;
        let mut command = ProcessCommand::new(executable);
        command
            .arg("--model")
            .arg(&args.model)
            .arg("--worker")
            .arg(match kind {
                WorkerKind::Pipeline => "pipeline",
                WorkerKind::Rank => "rank",
            })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(rank) = rank {
            command.arg("--rank").arg(rank.to_string());
        }
        if let Some(id) = id {
            command.arg("--nccl-id").arg(id);
        }
        let label = rank.map_or_else(|| "pipeline".to_owned(), |rank| format!("rank {rank}"));
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
            .with_context(|| format!("serialize command for {} worker", self.label))?;
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
            .with_context(|| format!("decode {} worker response", self.label))?
            .ensure_ok(&self.label)
    }

    fn receive_timed(&mut self) -> Result<WorkerResponse> {
        let mut poll_fd = libc::pollfd {
            fd: self.output.get_ref().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ready =
            unsafe { libc::poll(&mut poll_fd, 1, WORKER_RESPONSE_TIMEOUT.as_millis() as i32) };
        if ready < 0 {
            return Err(std::io::Error::last_os_error()).context("poll worker response");
        }
        if ready == 0 {
            self.terminate();
            bail!(
                "timed out after {} seconds waiting for {} worker",
                WORKER_RESPONSE_TIMEOUT.as_secs(),
                self.label
            )
        }
        self.receive()
    }

    fn request(&mut self, command: &WorkerCommand) -> Result<WorkerResponse> {
        self.send(command)?;
        self.receive_timed()
    }

    fn output_fd(&self) -> std::os::fd::RawFd {
        self.output.get_ref().as_raw_fd()
    }

    fn terminate(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
    }

    fn shutdown(mut self) -> Result<()> {
        let response = self.request(&WorkerCommand::Shutdown)?;
        anyhow::ensure!(response.status == "shutdown");
        let status = self.child.wait()?;
        anyhow::ensure!(
            status.success(),
            "{} worker exited with {status}",
            self.label
        );
        Ok(())
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

struct RunResult {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    prefill_worker_ms: f64,
    prefill_wall_ms: f64,
    decode_worker_ms: Vec<f64>,
    decode_wall_ms: Vec<f64>,
    load_memory: Vec<usize>,
    cache_memory: Vec<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.worker {
        Some(WorkerKind::Pipeline) => run_pipeline_worker(&args),
        Some(WorkerKind::Rank) => run_rank_worker(&args),
        None => run_controller(&args),
    }
}

fn run_controller(args: &Args) -> Result<()> {
    anyhow::ensure!(
        args.max_tokens > 0,
        "--max-tokens must be greater than zero"
    );
    anyhow::ensure!(
        (1..=2_048).contains(&args.prefill_chunk),
        "--prefill-chunk must be between 1 and 2048"
    );
    let shards = discover_gguf_shards(&args.model)?;
    let tokenizer = MiniMaxTokenizer::from_gguf(&shards[0])?;
    let messages: Vec<ChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": args.message}
    ]))?;
    let prompt = chat::render_prompt(&messages, &[], false)?;
    let prompt_ids = tokenizer.encode(&prompt)?;
    println!(
        "phase-5 full-model validation prompt_tokens={} max_tokens={} prefill_chunk={}",
        prompt_ids.len(),
        args.max_tokens,
        args.prefill_chunk
    );

    let pipeline = if args.skip_pipeline {
        None
    } else {
        let result = run_pipeline_controller(args, &prompt_ids)?;
        print_result("pipeline", &result, &tokenizer)?;
        Some(result)
    };
    let tensor = run_tensor_controller(args, &prompt_ids)?;
    print_result("tensor", &tensor, &tokenizer)?;

    if let Some(pipeline) = &pipeline {
        compare_values(
            "TP versus pipeline prefill logits",
            &pipeline.logits,
            &tensor.logits,
            None,
        )?;
        anyhow::ensure!(
            tensor.tokens == pipeline.tokens,
            "TP greedy tokens differ from pipeline: TP={:?}, pipeline={:?}",
            tensor.tokens,
            pipeline.tokens
        );
    }
    if args.message == "test" && args.max_tokens >= GOLDEN_TEST_TOKENS.len() {
        anyhow::ensure!(
            tensor.tokens[..GOLDEN_TEST_TOKENS.len()] == GOLDEN_TEST_TOKENS,
            "TP greedy golden mismatch: got {:?}",
            &tensor.tokens[..GOLDEN_TEST_TOKENS.len()]
        );
        if let Some(pipeline) = &pipeline {
            anyhow::ensure!(
                pipeline.tokens[..GOLDEN_TEST_TOKENS.len()] == GOLDEN_TEST_TOKENS,
                "pipeline greedy golden mismatch: got {:?}",
                &pipeline.tokens[..GOLDEN_TEST_TOKENS.len()]
            );
        }
        println!("llama.cpp 24-token greedy golden passed");
    }
    Ok(())
}

fn run_pipeline_controller(args: &Args, prompt_ids: &[u32]) -> Result<RunResult> {
    let mut worker = WorkerProcess::spawn(args, WorkerKind::Pipeline, None, None)?;
    let ready = worker.receive_timed()?;
    anyhow::ensure!(ready.status == "ready");
    let load_memory = ready.memory_used.unwrap_or_default();
    worker.request(&WorkerCommand::Reset)?;
    let (mut response, prefill_worker_ms, prefill_wall_ms) =
        pipeline_prefill(&mut worker, prompt_ids, args.prefill_chunk, true)?;
    let logits = response
        .logits
        .take()
        .ok_or_else(|| anyhow::anyhow!("pipeline returned no prefill logits"))?;
    let first_token = response
        .token_id
        .ok_or_else(|| anyhow::anyhow!("pipeline returned no greedy token"))?;
    let cache_memory = response.memory_used.unwrap_or_default();
    let mut tokens = vec![first_token];
    let mut decode_worker_ms = Vec::new();
    let mut decode_wall_ms = Vec::new();
    while tokens.len() < args.max_tokens {
        let pos = prompt_ids.len() + tokens.len() - 1;
        let started = Instant::now();
        response = worker.request(&WorkerCommand::Forward {
            ids: vec![*tokens.last().expect("generated token")],
            pos,
            return_logits: false,
            return_hidden: false,
            return_memory: false,
        })?;
        decode_wall_ms.push(started.elapsed().as_secs_f64() * 1e3);
        decode_worker_ms.push(response.milliseconds.context("pipeline decode timing")?);
        tokens.push(response.token_id.context("pipeline decode greedy token")?);
    }
    worker.shutdown()?;
    Ok(RunResult {
        tokens,
        logits,
        prefill_worker_ms,
        prefill_wall_ms,
        decode_worker_ms,
        decode_wall_ms,
        load_memory,
        cache_memory,
    })
}

fn pipeline_prefill(
    worker: &mut WorkerProcess,
    ids: &[u32],
    chunk_size: usize,
    return_logits: bool,
) -> Result<(WorkerResponse, f64, f64)> {
    let started = Instant::now();
    let mut worker_ms = 0.0;
    let mut pos = 0;
    let mut final_response = None;
    while pos < ids.len() {
        let end = (pos + chunk_size).min(ids.len());
        let response = worker.request(&WorkerCommand::Forward {
            ids: ids[pos..end].to_vec(),
            pos,
            return_logits: return_logits && end == ids.len(),
            return_hidden: false,
            return_memory: end == ids.len(),
        })?;
        anyhow::ensure!(response.cache_len == Some(end));
        worker_ms += response.milliseconds.context("pipeline prefill timing")?;
        final_response = Some(response);
        pos = end;
    }
    Ok((
        final_response.context("empty pipeline prefill")?,
        worker_ms,
        started.elapsed().as_secs_f64() * 1e3,
    ))
}

fn run_tensor_controller(args: &Args, prompt_ids: &[u32]) -> Result<RunResult> {
    let id = Id::new().context("create full-model process NCCL unique ID")?;
    let encoded_id = encode_nccl_id(&id);
    let mut ranks = [
        WorkerProcess::spawn(args, WorkerKind::Rank, Some(0), Some(&encoded_id))?,
        WorkerProcess::spawn(args, WorkerKind::Rank, Some(1), Some(&encoded_id))?,
    ];
    let ready = receive_pair(&mut ranks)?;
    anyhow::ensure!(ready[0].status == "ready" && ready[1].status == "ready");
    anyhow::ensure!(ready[0].rank == Some(0) && ready[1].rank == Some(1));
    println!(
        "TP ranks ready NCCL={} backend={} layer_weights=[{:.2}, {:.2}]GiB global_weights=[{:.2}, {:.2}]GiB",
        ready[0].nccl_version.unwrap_or_default(),
        ready[0].collective_backend.as_deref().unwrap_or("unknown"),
        gib(ready[0].layer_weight_bytes.unwrap_or_default()),
        gib(ready[1].layer_weight_bytes.unwrap_or_default()),
        gib(ready[0].global_weight_bytes.unwrap_or_default()),
        gib(ready[1].global_weight_bytes.unwrap_or_default()),
    );
    let load_memory = vec![
        ready[0]
            .memory_used
            .as_deref()
            .and_then(|memory| memory.first())
            .copied()
            .unwrap_or_default(),
        ready[1]
            .memory_used
            .as_deref()
            .and_then(|memory| memory.first())
            .copied()
            .unwrap_or_default(),
    ];

    if args.validate_prefill_lengths {
        validate_prefill_lengths(&mut ranks, args.prefill_chunk)?;
    }
    if args.check_transition {
        validate_transition(&mut ranks, args.prefill_chunk)?;
    }
    pair_request(&mut ranks, &WorkerCommand::Reset)?;
    let (mut response, prefill_worker_ms, prefill_wall_ms) =
        tensor_prefill(&mut ranks, prompt_ids, args.prefill_chunk, true, true, true)?;
    let logits = response[0]
        .logits
        .take()
        .ok_or_else(|| anyhow::anyhow!("rank 0 returned no prefill logits"))?;
    anyhow::ensure!(response[1].logits.is_none());
    compare_values(
        "full-model TP rank hidden agreement",
        response[0]
            .hidden
            .as_deref()
            .context("rank 0 returned no final hidden state")?,
        response[1]
            .hidden
            .as_deref()
            .context("rank 1 returned no final hidden state")?,
        Some(0.0),
    )?;
    let first_token = response[0]
        .token_id
        .context("rank 0 returned no greedy token")?;
    anyhow::ensure!(response[1].token_id.is_none());
    let cache_memory = vec![
        response[0]
            .memory_used
            .as_deref()
            .and_then(|memory| memory.first())
            .copied()
            .unwrap_or_default(),
        response[1]
            .memory_used
            .as_deref()
            .and_then(|memory| memory.first())
            .copied()
            .unwrap_or_default(),
    ];
    let mut tokens = vec![first_token];
    let mut decode_worker_ms = Vec::new();
    let mut decode_wall_ms = Vec::new();
    while tokens.len() < args.max_tokens {
        let pos = prompt_ids.len() + tokens.len() - 1;
        let started = Instant::now();
        response = pair_request(
            &mut ranks,
            &WorkerCommand::Forward {
                ids: vec![*tokens.last().expect("generated token")],
                pos,
                return_logits: false,
                return_hidden: false,
                return_memory: false,
            },
        )?;
        decode_wall_ms.push(started.elapsed().as_secs_f64() * 1e3);
        decode_worker_ms.push(
            response[0]
                .milliseconds
                .context("rank 0 decode timing")?
                .max(response[1].milliseconds.context("rank 1 decode timing")?),
        );
        let expected_cache = pos + 1;
        anyhow::ensure!(response[0].cache_len == Some(expected_cache));
        anyhow::ensure!(response[1].cache_len == Some(expected_cache));
        tokens.push(response[0].token_id.context("rank 0 decode greedy token")?);
    }
    validate_rewind(&mut ranks, prompt_ids, &tokens)?;

    pair_request(&mut ranks, &WorkerCommand::ClosePeerMapping)?;
    pair_request(&mut ranks, &WorkerCommand::FreeLocalPeerBuffer)?;
    let shutdown = pair_request(&mut ranks, &WorkerCommand::Shutdown)?;
    anyhow::ensure!(shutdown[0].status == "shutdown" && shutdown[1].status == "shutdown");
    let [mut rank0, mut rank1] = ranks;
    let status0 = rank0.child.wait()?;
    let status1 = rank1.child.wait()?;
    anyhow::ensure!(status0.success() && status1.success());

    Ok(RunResult {
        tokens,
        logits,
        prefill_worker_ms,
        prefill_wall_ms,
        decode_worker_ms,
        decode_wall_ms,
        load_memory,
        cache_memory,
    })
}

fn tensor_prefill(
    ranks: &mut [WorkerProcess; 2],
    ids: &[u32],
    chunk_size: usize,
    return_logits: bool,
    return_hidden: bool,
    return_memory: bool,
) -> Result<([WorkerResponse; 2], f64, f64)> {
    let started = Instant::now();
    let mut worker_ms = 0.0;
    let mut pos = 0;
    let mut final_response = None;
    while pos < ids.len() {
        let end = (pos + chunk_size).min(ids.len());
        let responses = pair_request(
            ranks,
            &WorkerCommand::Forward {
                ids: ids[pos..end].to_vec(),
                pos,
                return_logits: return_logits && end == ids.len(),
                return_hidden: return_hidden && end == ids.len(),
                return_memory: return_memory && end == ids.len(),
            },
        )?;
        anyhow::ensure!(responses[0].cache_len == Some(end));
        anyhow::ensure!(responses[1].cache_len == Some(end));
        worker_ms += responses[0]
            .milliseconds
            .context("rank 0 prefill timing")?
            .max(responses[1].milliseconds.context("rank 1 prefill timing")?);
        final_response = Some(responses);
        pos = end;
    }
    Ok((
        final_response.context("empty tensor prefill")?,
        worker_ms,
        started.elapsed().as_secs_f64() * 1e3,
    ))
}

fn validate_prefill_lengths(ranks: &mut [WorkerProcess; 2], chunk_size: usize) -> Result<()> {
    for length in [1usize, 5, 39, 512, 513] {
        pair_request(ranks, &WorkerCommand::Reset)?;
        let mut ids = vec![42u32; length];
        ids[0] = 1_000 + length as u32;
        let (responses, worker_ms, wall_ms) =
            tensor_prefill(ranks, &ids, chunk_size, false, false, false)?;
        let expected_shape = vec![1, LOCAL_KV_HEADS, length, HEAD_DIM];
        anyhow::ensure!(responses[0].cache_shape.as_ref() == Some(&expected_shape));
        anyhow::ensure!(responses[1].cache_shape.as_ref() == Some(&expected_shape));
        anyhow::ensure!(responses[0].token_id.is_some());
        println!("TP fresh prefill length={length} worker={worker_ms:.3}ms wall={wall_ms:.3}ms");
    }
    Ok(())
}

fn validate_transition(ranks: &mut [WorkerProcess; 2], chunk_size: usize) -> Result<()> {
    const TRANSITION: usize = 8_192;
    pair_request(ranks, &WorkerCommand::Reset)?;
    let mut ids = vec![42u32; TRANSITION - 1];
    ids[0] = 9_191;
    let (_, prefill_ms, wall_ms) = tensor_prefill(ranks, &ids, chunk_size, false, false, false)?;
    for pos in [TRANSITION - 1, TRANSITION] {
        let responses = pair_request(
            ranks,
            &WorkerCommand::Forward {
                ids: vec![43 + (pos + 1 - TRANSITION) as u32],
                pos,
                return_logits: false,
                return_hidden: true,
                return_memory: false,
            },
        )?;
        compare_values(
            &format!("TP rank hidden agreement at position {pos}"),
            responses[0]
                .hidden
                .as_deref()
                .context("rank 0 transition hidden")?,
            responses[1]
                .hidden
                .as_deref()
                .context("rank 1 transition hidden")?,
            Some(0.0),
        )?;
    }
    let expected_shape = vec![1, LOCAL_KV_HEADS, TRANSITION + 1, HEAD_DIM];
    let state = pair_request(
        ranks,
        &WorkerCommand::Truncate {
            seq_len: TRANSITION + 1,
        },
    )?;
    anyhow::ensure!(state[0].cache_shape.as_ref() == Some(&expected_shape));
    anyhow::ensure!(state[1].cache_shape.as_ref() == Some(&expected_shape));
    println!(
        "TP crossed the 8,192-token transition; prefill worker={prefill_ms:.3}ms wall={wall_ms:.3}ms"
    );
    Ok(())
}

fn validate_rewind(
    ranks: &mut [WorkerProcess; 2],
    prompt_ids: &[u32],
    generated: &[u32],
) -> Result<()> {
    let current_cache = prompt_ids.len() + generated.len() - 1;
    let (seq_len, replay_id) = if generated.len() >= 2 {
        (current_cache - 1, generated[generated.len() - 2])
    } else {
        (
            prompt_ids.len() - 1,
            *prompt_ids.last().context("empty prompt")?,
        )
    };
    let replay = |ranks: &mut [WorkerProcess; 2]| -> Result<[WorkerResponse; 2]> {
        pair_request(ranks, &WorkerCommand::Truncate { seq_len })?;
        pair_request(
            ranks,
            &WorkerCommand::Forward {
                ids: vec![replay_id],
                pos: seq_len,
                return_logits: true,
                return_hidden: true,
                return_memory: false,
            },
        )
    };
    let first = replay(ranks)?;
    let second = replay(ranks)?;
    compare_values(
        "TP cache rewind logits replay",
        first[0].logits.as_deref().context("first replay logits")?,
        second[0]
            .logits
            .as_deref()
            .context("second replay logits")?,
        Some(0.0),
    )?;
    anyhow::ensure!(first[0].token_id == second[0].token_id);
    compare_values(
        "TP cache rewind hidden replay rank 0",
        first[0].hidden.as_deref().context("first rank-0 hidden")?,
        second[0]
            .hidden
            .as_deref()
            .context("second rank-0 hidden")?,
        Some(0.0),
    )?;
    compare_values(
        "TP cache rewind hidden replay rank 1",
        first[1].hidden.as_deref().context("first rank-1 hidden")?,
        second[1]
            .hidden
            .as_deref()
            .context("second rank-1 hidden")?,
        Some(0.0),
    )?;
    println!("TP complete-model cache rewind replay passed at position {seq_len}");
    Ok(())
}

fn print_result(label: &str, result: &RunResult, tokenizer: &MiniMaxTokenizer) -> Result<()> {
    println!(
        "{label} memory load={} cache={} prefill worker={:.3}ms wall={:.3}ms",
        format_memory(&result.load_memory),
        format_memory(&result.cache_memory),
        result.prefill_worker_ms,
        result.prefill_wall_ms,
    );
    if !result.decode_worker_ms.is_empty() {
        println!(
            "{label} decode worker={:.3}ms ({:.1} tok/s) command-wall={:.3}ms ({:.1} tok/s)",
            mean(&result.decode_worker_ms),
            1_000.0 / mean(&result.decode_worker_ms),
            mean(&result.decode_wall_ms),
            1_000.0 / mean(&result.decode_wall_ms),
        );
    }
    if result.tokens.len() <= 64 {
        println!("{label} tokens={:?}", result.tokens);
        println!("{label} text={:?}", tokenizer.decode(&result.tokens)?);
    } else {
        println!(
            "{label} tokens={} first={:?} last={:?}",
            result.tokens.len(),
            &result.tokens[..24],
            &result.tokens[result.tokens.len() - 8..]
        );
        println!(
            "{label} first-24 text={:?}",
            tokenizer.decode(&result.tokens[..24])?
        );
    }
    Ok(())
}

fn run_pipeline_worker(args: &Args) -> Result<()> {
    let shards = discover_gguf_shards(&args.model)?;
    let devices = vec![Device::new_cuda(0)?, Device::new_cuda(1)?];
    for device in &devices {
        unsafe { device.as_cuda_device()?.disable_event_tracking() };
    }
    let mut model = PipelineModel::load(&shards, &devices)?;
    synchronize_devices(&devices)?;
    write_response(&WorkerResponse::ready(
        None,
        None,
        None,
        None,
        None,
        device_memory_used(&devices)?,
    ))?;
    let mut cache_len = 0usize;
    worker_loop(|command| match command {
        WorkerCommand::Reset => {
            model.reset();
            cache_len = 0;
            Ok(WorkerResponse::done(0, None))
        }
        WorkerCommand::Forward {
            ids,
            pos,
            return_logits,
            return_hidden,
            return_memory,
        } => {
            anyhow::ensure!(
                !return_hidden,
                "pipeline worker cannot return final hidden state"
            );
            anyhow::ensure!(
                cache_len == pos,
                "pipeline cache is {cache_len}, command position is {pos}"
            );
            let started = Instant::now();
            let logits = model.forward(&ids, pos)?;
            let token_id = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            devices[0].synchronize()?;
            let milliseconds = started.elapsed().as_secs_f64() * 1e3;
            cache_len += ids.len();
            let mut response = WorkerResponse::done(cache_len, None);
            response.token_id = Some(token_id);
            response.milliseconds = Some(milliseconds);
            if return_memory {
                response.memory_used = Some(device_memory_used(&devices)?);
            }
            if return_logits {
                response.logits = Some(logits.to_vec1::<f32>()?);
            }
            Ok(response)
        }
        WorkerCommand::Truncate { seq_len } => {
            model.truncate_cache(seq_len)?;
            cache_len = seq_len;
            Ok(WorkerResponse::done(cache_len, None))
        }
        WorkerCommand::ClosePeerMapping | WorkerCommand::FreeLocalPeerBuffer => {
            bail!("pipeline worker received a rank-only shutdown command")
        }
        WorkerCommand::Shutdown => unreachable!("shutdown handled by worker loop"),
    })
}

fn run_rank_worker(args: &Args) -> Result<()> {
    let rank = args.rank.context("rank worker requires --rank")?;
    anyhow::ensure!(rank < 2, "rank worker requires rank 0 or 1");
    let id = decode_nccl_id(
        args.nccl_id
            .as_deref()
            .context("rank worker requires --nccl-id")?,
    )?;
    let shards = discover_gguf_shards(&args.model)?;
    let device = Device::new_cuda(rank)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let mut model = TensorParallelRankModel::load(&shards, rank, &device, id)?;
    device.synchronize()?;
    write_response(&WorkerResponse::ready(
        Some(rank),
        Some(model.layer_weight_bytes()),
        Some(model.global_weight_bytes()),
        Some(model.nccl_version()),
        Some(model.collective_backend().to_owned()),
        device_memory_used(std::slice::from_ref(&device))?,
    ))?;
    worker_loop(|command| match command {
        WorkerCommand::Reset => {
            model.reset();
            Ok(WorkerResponse::done(0, None))
        }
        WorkerCommand::Forward {
            ids,
            pos,
            return_logits,
            return_hidden,
            return_memory,
        } => {
            let started = Instant::now();
            let output = model.forward(&ids, pos)?;
            let token_id = output
                .logits
                .as_ref()
                .map(|logits| logits.argmax(D::Minus1)?.to_scalar::<u32>())
                .transpose()?;
            device.synchronize()?;
            let milliseconds = started.elapsed().as_secs_f64() * 1e3;
            let cache_len = model.cache_length()?;
            let mut response = WorkerResponse::done(cache_len, model.cache_shape()?);
            response.token_id = token_id;
            response.milliseconds = Some(milliseconds);
            if return_memory {
                response.memory_used = Some(device_memory_used(std::slice::from_ref(&device))?);
            }
            if return_logits {
                response.logits = output
                    .logits
                    .as_ref()
                    .map(Tensor::to_vec1::<f32>)
                    .transpose()?;
            }
            if return_hidden {
                response.hidden = Some(output.hidden.flatten_all()?.to_vec1::<f32>()?);
            }
            Ok(response)
        }
        WorkerCommand::Truncate { seq_len } => {
            model.truncate_cache(seq_len)?;
            Ok(WorkerResponse::done(seq_len, model.cache_shape()?))
        }
        WorkerCommand::ClosePeerMapping => {
            model.close_peer_mapping()?;
            Ok(WorkerResponse::done(
                model.cache_length()?,
                model.cache_shape()?,
            ))
        }
        WorkerCommand::FreeLocalPeerBuffer => {
            model.free_local_peer_buffer()?;
            Ok(WorkerResponse::done(
                model.cache_length()?,
                model.cache_shape()?,
            ))
        }
        WorkerCommand::Shutdown => unreachable!("shutdown handled by worker loop"),
    })
}

fn worker_loop(mut handle: impl FnMut(WorkerCommand) -> Result<WorkerResponse>) -> Result<()> {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let command: WorkerCommand =
            serde_json::from_str(&line?).context("decode worker command")?;
        if matches!(command, WorkerCommand::Shutdown) {
            let mut response = WorkerResponse::done(0, None);
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
    bail!("worker command pipe closed without shutdown")
}

fn write_response(response: &WorkerResponse) -> Result<()> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, response)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn terminate_pair(ranks: &mut [WorkerProcess; 2]) {
    for rank in ranks.iter_mut() {
        rank.terminate();
    }
    for rank in ranks.iter_mut() {
        let _ = rank.child.wait();
    }
}

fn receive_pair(ranks: &mut [WorkerProcess; 2]) -> Result<[WorkerResponse; 2]> {
    let deadline = Instant::now() + WORKER_RESPONSE_TIMEOUT;
    let mut responses = [None, None];
    while responses.iter().any(Option::is_none) {
        let now = Instant::now();
        if now >= deadline {
            terminate_pair(ranks);
            bail!("timed out waiting for rank workers")
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
        let ready = unsafe { libc::poll(poll_fds.as_mut_ptr(), 2, timeout_ms) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            terminate_pair(ranks);
            return Err(error).context("poll rank worker responses");
        }
        if ready == 0 {
            terminate_pair(ranks);
            bail!("timed out waiting for rank workers")
        }
        for rank in 0..2 {
            let terminal = libc::POLLIN | libc::POLLHUP | libc::POLLERR | libc::POLLNVAL;
            if responses[rank].is_none() && poll_fds[rank].revents & terminal != 0 {
                match ranks[rank].receive() {
                    Ok(response) => responses[rank] = Some(response),
                    Err(error) => {
                        terminate_pair(ranks);
                        return Err(error).with_context(|| format!("receive rank {rank} response"));
                    }
                }
            }
        }
    }
    let [Some(rank0), Some(rank1)] = responses else {
        unreachable!("both rank responses were populated")
    };
    Ok([rank0, rank1])
}

fn pair_request(
    ranks: &mut [WorkerProcess; 2],
    command: &WorkerCommand,
) -> Result<[WorkerResponse; 2]> {
    if let Err(error) = ranks[0].send(command).and_then(|()| ranks[1].send(command)) {
        terminate_pair(ranks);
        return Err(error).context("send paired worker command");
    }
    receive_pair(ranks)
}

fn synchronize_devices(devices: &[Device]) -> Result<()> {
    for device in devices {
        device.synchronize()?;
    }
    Ok(())
}

fn device_memory_used(devices: &[Device]) -> Result<Vec<usize>> {
    devices
        .iter()
        .map(|device| {
            let context = device.as_cuda_device()?.cuda_stream().context().clone();
            context.bind_to_thread()?;
            let (free, total) = context.mem_get_info()?;
            Ok(total.saturating_sub(free))
        })
        .collect()
}

fn compare_values(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    ceiling: Option<f32>,
) -> Result<()> {
    anyhow::ensure!(
        expected.len() == actual.len(),
        "{label} length mismatch: {} vs {}",
        expected.len(),
        actual.len()
    );
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (&expected, &actual) in expected.iter().zip(actual) {
        let difference = (actual - expected).abs();
        max_abs = max_abs.max(difference);
        sum_abs += difference as f64;
    }
    let mean_abs = sum_abs / expected.len() as f64;
    anyhow::ensure!(
        max_abs.is_finite() && mean_abs.is_finite(),
        "{label} produced a non-finite difference"
    );
    println!("{label}: max_abs={max_abs:.7} mean_abs={mean_abs:.7}");
    if let Some(ceiling) = ceiling {
        anyhow::ensure!(
            max_abs <= ceiling,
            "{label} max_abs={max_abs} exceeds {ceiling}"
        );
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

fn format_memory(memory: &[usize]) -> String {
    let values = memory
        .iter()
        .map(|&bytes| format!("{:.2}GiB", gib(bytes)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
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
    anyhow::ensure!(
        encoded.len() == 256,
        "NCCL ID must contain 256 hexadecimal characters"
    );
    let mut internal = [0 as c_char; 128];
    for (index, byte) in internal.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16)
            .with_context(|| format!("decode NCCL ID byte {index}"))? as c_char;
    }
    Ok(Id::uninit(internal))
}
