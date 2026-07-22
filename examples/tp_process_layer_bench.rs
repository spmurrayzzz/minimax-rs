use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use clap::{Parser, ValueEnum};
use cudarc::nccl::Id;
use minimax::{
    model_files::discover_gguf_shards,
    tensor_parallel::TensorParallelRankGroup,
    tensor_parallel_model::{
        HEAD_DIM, HIDDEN_SIZE, LOCAL_KV_HEADS, TensorParallelRankLayer, rotary_embedding,
    },
    tp_reference::FullLayerReference,
};
use serde::{Deserialize, Serialize};
use std::{
    ffi::c_char,
    io::{BufRead, BufReader, BufWriter, Write},
    os::fd::AsRawFd,
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command as ProcessCommand, Stdio},
    time::{Duration, Instant},
};

const PREFILL_CHUNK: usize = 512;
const PREFILL_SEED: u64 = 0x4d49_4e49_4d41_5800;
const DECODE_SEED: u64 = 0x5450_325f_4445_434f;
const WORKER_RESPONSE_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone, Copy, Debug, ValueEnum)]
enum WorkerKind {
    Reference,
    Rank,
}

#[derive(Debug, Parser)]
struct Args {
    /// Directory containing the four split GGUF files.
    #[arg(long, env = "MINIMAX_MODEL_DIR", value_name = "DIR")]
    model: PathBuf,
    #[arg(long, default_value_t = 0)]
    layer: usize,
    /// Number of consecutive layers to include in the process/graph slice.
    #[arg(long, default_value_t = 1)]
    layers: usize,
    #[arg(long, default_value_t = 512)]
    context: usize,
    #[arg(long, default_value_t = 1_000)]
    iterations: usize,
    #[arg(long, default_value_t = 50)]
    warmup: usize,
    /// Also validate cached decode immediately before and after position 8,192.
    #[arg(long)]
    check_transition: bool,
    /// Capture one fixed-position decode graph per worker and benchmark replay.
    #[arg(long)]
    cuda_graphs: bool,
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
        query_len: usize,
        pos: usize,
        prefill: bool,
        seed: u64,
        broadcast: bool,
        return_output: bool,
    },
    Benchmark {
        iterations: usize,
        warmup: usize,
        start_pos: usize,
        seed: u64,
        broadcast: bool,
    },
    GraphBenchmark {
        iterations: usize,
        warmup: usize,
        pos: usize,
        seed: u64,
        broadcast: bool,
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
    weight_bytes: Option<usize>,
    nccl_version: Option<i32>,
    cache_len: Option<usize>,
    cache_shape: Option<Vec<usize>>,
    output: Option<Vec<f32>>,
    milliseconds: Option<f64>,
    message: Option<String>,
}

impl WorkerResponse {
    fn ready(rank: Option<usize>, weight_bytes: usize, nccl_version: Option<i32>) -> Self {
        Self {
            status: "ready".to_string(),
            rank,
            weight_bytes: Some(weight_bytes),
            nccl_version,
            cache_len: Some(0),
            cache_shape: None,
            output: None,
            milliseconds: None,
            message: None,
        }
    }

    fn done(cache_len: usize, cache_shape: Option<Vec<usize>>) -> Self {
        Self {
            status: "done".to_string(),
            rank: None,
            weight_bytes: None,
            nccl_version: None,
            cache_len: Some(cache_len),
            cache_shape,
            output: None,
            milliseconds: None,
            message: None,
        }
    }

    fn error(error: &anyhow::Error) -> Self {
        Self {
            status: "error".to_string(),
            rank: None,
            weight_bytes: None,
            nccl_version: None,
            cache_len: None,
            cache_shape: None,
            output: None,
            milliseconds: None,
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
        let executable = std::env::current_exe().context("resolve process benchmark executable")?;
        let mut command = ProcessCommand::new(executable);
        command
            .arg("--model")
            .arg(&args.model)
            .arg("--layer")
            .arg(args.layer.to_string())
            .arg("--layers")
            .arg(args.layers.to_string())
            .arg("--worker")
            .arg(match kind {
                WorkerKind::Reference => "reference",
                WorkerKind::Rank => "rank",
            })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if args.cuda_graphs {
            command.arg("--cuda-graphs");
        }
        if let Some(rank) = rank {
            command.arg("--rank").arg(rank.to_string());
        }
        if let Some(id) = id {
            command.arg("--nccl-id").arg(id);
        }
        let label = match rank {
            Some(rank) => format!("rank {rank}"),
            None => "reference".to_string(),
        };
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
            .with_context(|| format!("decode {} worker response: {line:?}", self.label))?
            .ensure_ok(&self.label)
    }

    fn request(&mut self, command: &WorkerCommand) -> Result<WorkerResponse> {
        self.send(command)?;
        self.receive()
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
        self.send(&WorkerCommand::Shutdown)?;
        let response = self.receive()?;
        if response.status != "shutdown" {
            bail!(
                "{} worker returned {} while shutting down",
                self.label,
                response.status
            )
        }
        let status = self.child.wait()?;
        if !status.success() {
            bail!("{} worker exited with {status}", self.label)
        }
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

fn main() -> Result<()> {
    let args = Args::parse();
    match args.worker {
        Some(WorkerKind::Reference) => run_reference_worker(&args),
        Some(WorkerKind::Rank) => run_rank_worker(&args),
        None => run_controller(&args),
    }
}

fn run_controller(args: &Args) -> Result<()> {
    anyhow::ensure!(args.context > 0, "--context must be greater than zero");
    anyhow::ensure!(args.layers > 0, "--layers must be greater than zero");
    anyhow::ensure!(
        args.layer + args.layers <= 62,
        "requested layer range exceeds MiniMax's 62 layers"
    );
    anyhow::ensure!(
        args.iterations > 0,
        "--iterations must be greater than zero"
    );
    anyhow::ensure!(
        args.context + args.warmup + args.iterations < 196_608,
        "context plus benchmark iterations exceeds the model context limit"
    );

    println!(
        "phase-4b process benchmark layers={}..{} context={} warmup={} iterations={}",
        args.layer,
        args.layer + args.layers,
        args.context,
        args.warmup,
        args.iterations
    );
    let (reference_output, reference_ms, reference_graph_ms, reference_bytes) =
        run_reference_controller(args)?;
    let id = Id::new().context("create process NCCL unique ID")?;
    let encoded_id = encode_nccl_id(&id);
    let mut ranks = [
        WorkerProcess::spawn(args, WorkerKind::Rank, Some(0), Some(&encoded_id))?,
        WorkerProcess::spawn(args, WorkerKind::Rank, Some(1), Some(&encoded_id))?,
    ];
    let [ready0, ready1] = receive_pair(&mut ranks)?;
    anyhow::ensure!(ready0.status == "ready" && ready1.status == "ready");
    anyhow::ensure!(ready0.rank == Some(0) && ready1.rank == Some(1));
    println!(
        "rank workers ready NCCL={} weights=[{:.2}, {:.2}]MiB reference={:.2}MiB",
        ready0.nccl_version.unwrap_or_default(),
        ready0.weight_bytes.unwrap_or_default() as f64 / 1_048_576.0,
        ready1.weight_bytes.unwrap_or_default() as f64 / 1_048_576.0,
        reference_bytes as f64 / 1_048_576.0,
    );

    pair_reset(&mut ranks)?;
    pair_prefill(&mut ranks, args.context)?;
    let decode = WorkerCommand::Forward {
        query_len: 1,
        pos: args.context,
        prefill: false,
        seed: DECODE_SEED,
        broadcast: true,
        return_output: true,
    };
    let [rank0, rank1] = pair_request(&mut ranks, &decode)?;
    let output0 = rank0
        .output
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("rank 0 returned no parity output"))?;
    let output1 = rank1
        .output
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("rank 1 returned no parity output"))?;
    compare_values("process TP rank agreement", output0, output1, 0.0)?;
    compare_values(
        "process TP versus full reference",
        &reference_output,
        output0,
        4.0 * args.layers as f32,
    )?;
    let expected_shape = vec![1, LOCAL_KV_HEADS, args.context + 1, HEAD_DIM];
    anyhow::ensure!(rank0.cache_shape.as_ref() == Some(&expected_shape));
    anyhow::ensure!(rank1.cache_shape.as_ref() == Some(&expected_shape));

    let truncate = WorkerCommand::Truncate {
        seq_len: args.context,
    };
    pair_request(&mut ranks, &truncate)?;
    let replay = pair_request(&mut ranks, &decode)?;
    compare_values(
        "rank 0 cache rewind replay",
        output0,
        replay[0]
            .output
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("rank 0 returned no replay output"))?,
        0.0,
    )?;
    compare_values(
        "rank 1 cache rewind replay",
        output1,
        replay[1]
            .output
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("rank 1 returned no replay output"))?,
        0.0,
    )?;

    if args.check_transition {
        validate_transition(&mut ranks)?;
    }

    let tensor_no_broadcast = benchmark_pair(args, &mut ranks, false)?;
    let tensor_with_broadcast = benchmark_pair(args, &mut ranks, true)?;
    println!(
        "decode reference={reference_ms:.3}ms process-TP/IPC={tensor_no_broadcast:.3}ms ({:.3}x) process-TP+broadcast={tensor_with_broadcast:.3}ms ({:.3}x)",
        reference_ms / tensor_no_broadcast,
        reference_ms / tensor_with_broadcast,
    );
    if args.cuda_graphs {
        let (reference_graph_ms, reference_graph_label) = match reference_graph_ms {
            Some(milliseconds) => (milliseconds, "CUDA-graph"),
            None => (reference_ms, "eager"),
        };
        let (tensor_graph, graph_output) = graph_benchmark_pair(args, &mut ranks, false)?;
        compare_values(
            "CUDA graph replay versus eager TP",
            output0,
            &graph_output,
            0.0,
        )?;
        let (tensor_graph_broadcast, graph_broadcast_output) =
            graph_benchmark_pair(args, &mut ranks, true)?;
        compare_values(
            "CUDA graph broadcast replay versus eager TP",
            output0,
            &graph_broadcast_output,
            0.0,
        )?;
        println!(
            "CUDA graph fixed-pos {reference_graph_label}-reference={reference_graph_ms:.3}ms process-TP/IPC={tensor_graph:.3}ms ({:.3}x) process-TP+broadcast={tensor_graph_broadcast:.3}ms ({:.3}x)",
            reference_graph_ms / tensor_graph,
            reference_graph_ms / tensor_graph_broadcast,
        );
    }

    // CUDA requires every importer to close its IPC mapping before either
    // exporter frees the underlying allocation.
    pair_request(&mut ranks, &WorkerCommand::ClosePeerMapping)?;
    pair_request(&mut ranks, &WorkerCommand::FreeLocalPeerBuffer)?;
    let shutdown = pair_request(&mut ranks, &WorkerCommand::Shutdown)?;
    anyhow::ensure!(shutdown[0].status == "shutdown" && shutdown[1].status == "shutdown");
    let [mut rank0, mut rank1] = ranks;
    let status0 = rank0.child.wait()?;
    let status1 = rank1.child.wait()?;
    anyhow::ensure!(status0.success() && status1.success());
    Ok(())
}

fn run_reference_controller(args: &Args) -> Result<(Vec<f32>, f64, Option<f64>, usize)> {
    let mut worker = WorkerProcess::spawn(args, WorkerKind::Reference, None, None)?;
    let ready = worker.receive()?;
    anyhow::ensure!(ready.status == "ready");
    let weight_bytes = ready.weight_bytes.unwrap_or_default();
    worker.request(&WorkerCommand::Reset)?;
    reference_prefill(&mut worker, args.context)?;
    let response = worker.request(&WorkerCommand::Forward {
        query_len: 1,
        pos: args.context,
        prefill: false,
        seed: DECODE_SEED,
        broadcast: false,
        return_output: true,
    })?;
    let output = response
        .output
        .ok_or_else(|| anyhow::anyhow!("reference worker returned no parity output"))?;

    worker.request(&WorkerCommand::Reset)?;
    reference_prefill(&mut worker, args.context)?;
    let started = Instant::now();
    let response = worker.request(&WorkerCommand::Benchmark {
        iterations: args.iterations,
        warmup: args.warmup,
        start_pos: args.context,
        seed: DECODE_SEED,
        broadcast: false,
    })?;
    let command_ms = started.elapsed().as_secs_f64() * 1e3 / args.iterations as f64;
    let worker_ms = response
        .milliseconds
        .ok_or_else(|| anyhow::anyhow!("reference worker returned no timing"))?;
    println!("reference worker={worker_ms:.3}ms command-wall={command_ms:.3}ms");
    let graph_ms = if args.cuda_graphs && args.layers == 1 {
        worker.request(&WorkerCommand::Reset)?;
        reference_prefill(&mut worker, args.context)?;
        let started = Instant::now();
        let response = worker.request(&WorkerCommand::GraphBenchmark {
            iterations: args.iterations,
            warmup: args.warmup,
            pos: args.context,
            seed: DECODE_SEED,
            broadcast: false,
        })?;
        let command_ms = started.elapsed().as_secs_f64() * 1e3 / args.iterations as f64;
        let worker_graph_ms = response
            .milliseconds
            .ok_or_else(|| anyhow::anyhow!("reference worker returned no CUDA graph timing"))?;
        println!(
            "reference CUDA graph worker={worker_graph_ms:.3}ms command-wall={command_ms:.3}ms"
        );
        Some(worker_graph_ms)
    } else {
        None
    };
    worker.shutdown()?;
    Ok((output, worker_ms, graph_ms, weight_bytes))
}

fn benchmark_pair(args: &Args, ranks: &mut [WorkerProcess; 2], broadcast: bool) -> Result<f64> {
    pair_reset(ranks)?;
    pair_prefill(ranks, args.context)?;
    let command = WorkerCommand::Benchmark {
        iterations: args.iterations,
        warmup: args.warmup,
        start_pos: args.context,
        seed: DECODE_SEED,
        broadcast,
    };
    let started = Instant::now();
    let responses = pair_request(ranks, &command)?;
    let command_ms = started.elapsed().as_secs_f64() * 1e3 / args.iterations as f64;
    let rank0_ms = responses[0]
        .milliseconds
        .ok_or_else(|| anyhow::anyhow!("rank 0 returned no timing"))?;
    let rank1_ms = responses[1]
        .milliseconds
        .ok_or_else(|| anyhow::anyhow!("rank 1 returned no timing"))?;
    let worker_ms = rank0_ms.max(rank1_ms);
    println!(
        "process TP broadcast={broadcast} rank-worker=[{rank0_ms:.3}, {rank1_ms:.3}]ms command-wall={command_ms:.3}ms"
    );
    Ok(worker_ms)
}

fn graph_benchmark_pair(
    args: &Args,
    ranks: &mut [WorkerProcess; 2],
    broadcast: bool,
) -> Result<(f64, Vec<f32>)> {
    pair_reset(ranks)?;
    pair_prefill(ranks, args.context)?;
    let command = WorkerCommand::GraphBenchmark {
        iterations: args.iterations,
        warmup: args.warmup,
        pos: args.context,
        seed: DECODE_SEED,
        broadcast,
    };
    let started = Instant::now();
    let responses = pair_request(ranks, &command)?;
    let command_ms = started.elapsed().as_secs_f64() * 1e3 / args.iterations as f64;
    let rank0_ms = responses[0]
        .milliseconds
        .ok_or_else(|| anyhow::anyhow!("rank 0 returned no CUDA graph timing"))?;
    let rank1_ms = responses[1]
        .milliseconds
        .ok_or_else(|| anyhow::anyhow!("rank 1 returned no CUDA graph timing"))?;
    let worker_ms = rank0_ms.max(rank1_ms);
    println!(
        "process TP CUDA graph broadcast={broadcast} rank-worker=[{rank0_ms:.3}, {rank1_ms:.3}]ms command-wall={command_ms:.3}ms"
    );
    let output0 = responses[0]
        .output
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("rank 0 returned no CUDA graph output"))?;
    let output1 = responses[1]
        .output
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("rank 1 returned no CUDA graph output"))?;
    compare_values("CUDA graph process rank agreement", output0, output1, 0.0)?;
    Ok((worker_ms, output0.to_vec()))
}

fn reference_prefill(worker: &mut WorkerProcess, context: usize) -> Result<()> {
    let mut pos = 0;
    while pos < context {
        let query_len = (context - pos).min(PREFILL_CHUNK);
        let response = worker.request(&WorkerCommand::Forward {
            query_len,
            pos,
            prefill: true,
            seed: PREFILL_SEED.wrapping_add(pos as u64),
            broadcast: false,
            return_output: false,
        })?;
        anyhow::ensure!(response.cache_len == Some(pos + query_len));
        pos += query_len;
    }
    Ok(())
}

fn pair_prefill(ranks: &mut [WorkerProcess; 2], context: usize) -> Result<()> {
    let mut pos = 0;
    while pos < context {
        let query_len = (context - pos).min(PREFILL_CHUNK);
        let responses = pair_request(
            ranks,
            &WorkerCommand::Forward {
                query_len,
                pos,
                prefill: true,
                seed: PREFILL_SEED.wrapping_add(pos as u64),
                broadcast: true,
                return_output: false,
            },
        )?;
        anyhow::ensure!(responses[0].cache_len == Some(pos + query_len));
        anyhow::ensure!(responses[1].cache_len == Some(pos + query_len));
        pos += query_len;
    }
    Ok(())
}

fn pair_reset(ranks: &mut [WorkerProcess; 2]) -> Result<()> {
    let responses = pair_request(ranks, &WorkerCommand::Reset)?;
    anyhow::ensure!(responses[0].cache_len == Some(0));
    anyhow::ensure!(responses[1].cache_len == Some(0));
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
            bail!(
                "timed out after {} seconds waiting for rank workers",
                WORKER_RESPONSE_TIMEOUT.as_secs()
            )
        }
        let remaining = deadline.saturating_duration_since(now);
        let timeout_ms = remaining.as_millis().min(i32::MAX as u128) as i32;
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
        let ready = unsafe { libc::poll(poll_fds.as_mut_ptr(), poll_fds.len() as _, timeout_ms) };
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
            bail!(
                "timed out after {} seconds waiting for rank workers",
                WORKER_RESPONSE_TIMEOUT.as_secs()
            )
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

fn validate_transition(ranks: &mut [WorkerProcess; 2]) -> Result<()> {
    const TRANSITION: usize = 8_192;
    pair_reset(ranks)?;
    pair_prefill(ranks, TRANSITION - 1)?;
    for pos in [TRANSITION - 1, TRANSITION] {
        let responses = pair_request(
            ranks,
            &WorkerCommand::Forward {
                query_len: 1,
                pos,
                prefill: false,
                seed: DECODE_SEED.wrapping_add(pos as u64),
                broadcast: true,
                return_output: true,
            },
        )?;
        compare_values(
            &format!("process TP rank agreement at position {pos}"),
            responses[0]
                .output
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("rank 0 returned no transition output"))?,
            responses[1]
                .output
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("rank 1 returned no transition output"))?,
            0.0,
        )?;
    }
    println!("process TP decode crossed the 8,192-token attention transition");
    Ok(())
}

fn reference_forward(
    layers: &mut [FullLayerReference],
    mut hidden: Tensor,
    pos: usize,
    prefill: bool,
) -> Result<Tensor> {
    for (offset, layer) in layers.iter_mut().enumerate() {
        hidden = layer
            .forward(&hidden, pos, prefill)
            .with_context(|| format!("reference layer offset {offset}"))?;
    }
    Ok(hidden)
}

fn reference_cache_state(layers: &[FullLayerReference]) -> Result<(usize, Option<Vec<usize>>)> {
    let first = layers
        .first()
        .ok_or_else(|| anyhow::anyhow!("reference worker has no layers"))?;
    let length = first.cache_length();
    for (offset, layer) in layers.iter().enumerate().skip(1) {
        anyhow::ensure!(
            layer.cache_length() == length,
            "reference layer offset {offset} cache length {} differs from {length}",
            layer.cache_length()
        );
    }
    Ok((length, first.cache_shape()?))
}

fn rank_forward(
    layers: &mut [TensorParallelRankLayer],
    mut hidden: Tensor,
    pos: usize,
    prefill: bool,
    collective: &mut TensorParallelRankGroup,
) -> Result<Tensor> {
    for layer in layers {
        hidden = layer.forward(hidden, pos, prefill, collective)?;
    }
    Ok(hidden)
}

fn rank_cache_state(layers: &[TensorParallelRankLayer]) -> Result<(usize, Option<Vec<usize>>)> {
    let first = layers
        .first()
        .ok_or_else(|| anyhow::anyhow!("rank worker has no layers"))?;
    let length = first.cache_length();
    for (offset, layer) in layers.iter().enumerate().skip(1) {
        anyhow::ensure!(
            layer.cache_length() == length,
            "rank layer offset {offset} cache length {} differs from {length}",
            layer.cache_length()
        );
    }
    Ok((length, first.cache_shape()?))
}

fn run_reference_worker(args: &Args) -> Result<()> {
    let shards = discover_gguf_shards(&args.model)?;
    let device = Device::new_cuda(0)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let _graph_cache_guard = args.cuda_graphs.then(|| {
        device
            .as_cuda_device()
            .expect("CUDA device")
            .enable_cuda_graph_htod_cache()
    });
    let rope = rotary_embedding(&device)?;
    let mut layers = (args.layer..args.layer + args.layers)
        .map(|layer| FullLayerReference::load(&shards, layer, &device, rope.clone()))
        .collect::<Result<Vec<_>>>()?;
    let weight_bytes = layers.iter().map(FullLayerReference::weight_bytes).sum();
    write_response(&WorkerResponse::ready(None, weight_bytes, None))?;
    worker_loop(|command| match command {
        WorkerCommand::Reset => {
            for layer in &mut layers {
                layer.reset();
            }
            Ok(WorkerResponse::done(0, None))
        }
        WorkerCommand::Forward {
            query_len,
            pos,
            prefill,
            seed,
            return_output,
            ..
        } => {
            anyhow::ensure!(reference_cache_state(&layers)?.0 == pos);
            let input = deterministic_tensor(&device, query_len, seed, false)?;
            let output = reference_forward(&mut layers, input, pos, prefill)?;
            device.synchronize()?;
            let (cache_len, cache_shape) = reference_cache_state(&layers)?;
            let mut response = WorkerResponse::done(cache_len, cache_shape);
            if return_output {
                response.output = Some(output.flatten_all()?.to_vec1::<f32>()?);
            }
            Ok(response)
        }
        WorkerCommand::Benchmark {
            iterations,
            warmup,
            start_pos,
            seed,
            ..
        } => {
            anyhow::ensure!(reference_cache_state(&layers)?.0 == start_pos);
            let input = deterministic_tensor(&device, 1, seed, false)?;
            device.synchronize()?;
            for iteration in 0..warmup {
                drop(reference_forward(
                    &mut layers,
                    input.clone(),
                    start_pos + iteration,
                    false,
                )?);
            }
            device.synchronize()?;
            let started = Instant::now();
            for iteration in 0..iterations {
                drop(reference_forward(
                    &mut layers,
                    input.clone(),
                    start_pos + warmup + iteration,
                    false,
                )?);
            }
            device.synchronize()?;
            let (cache_len, cache_shape) = reference_cache_state(&layers)?;
            let mut response = WorkerResponse::done(cache_len, cache_shape);
            response.milliseconds = Some(started.elapsed().as_secs_f64() * 1e3 / iterations as f64);
            Ok(response)
        }
        WorkerCommand::GraphBenchmark {
            iterations,
            warmup,
            pos,
            seed,
            ..
        } => {
            anyhow::ensure!(reference_cache_state(&layers)?.0 == pos);
            let input = deterministic_tensor(&device, 1, seed, false)?;
            device.synchronize()?;
            let stream = device.as_cuda_device()?.cuda_stream();
            stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                )
                .context("begin reference CUDA graph capture")?;
            drop(
                reference_forward(&mut layers, input, pos, false)
                    .context("enqueue reference CUDA graph")?,
            );
            let graph = stream
                .end_capture(
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .context("end and instantiate reference CUDA graph")?
                .ok_or_else(|| anyhow::anyhow!("reference CUDA graph capture returned no graph"))?;
            graph.upload().context("upload reference CUDA graph")?;
            for _ in 0..warmup {
                graph.launch().context("warm reference CUDA graph")?;
            }
            device.synchronize()?;
            let started = Instant::now();
            for _ in 0..iterations {
                graph.launch().context("benchmark reference CUDA graph")?;
            }
            device.synchronize()?;
            let (cache_len, cache_shape) = reference_cache_state(&layers)?;
            let mut response = WorkerResponse::done(cache_len, cache_shape);
            response.milliseconds = Some(started.elapsed().as_secs_f64() * 1e3 / iterations as f64);
            Ok(response)
        }
        WorkerCommand::Truncate { seq_len } => {
            for layer in &mut layers {
                layer.truncate_cache(seq_len)?;
            }
            let (cache_len, cache_shape) = reference_cache_state(&layers)?;
            Ok(WorkerResponse::done(cache_len, cache_shape))
        }
        WorkerCommand::ClosePeerMapping | WorkerCommand::FreeLocalPeerBuffer => {
            bail!("reference worker received a rank-only shutdown command")
        }
        WorkerCommand::Shutdown => unreachable!("shutdown handled by worker loop"),
    })
}

fn run_rank_worker(args: &Args) -> Result<()> {
    let rank = args
        .rank
        .ok_or_else(|| anyhow::anyhow!("rank worker requires --rank"))?;
    anyhow::ensure!(rank < 2, "rank worker requires rank 0 or 1");
    let id = decode_nccl_id(
        args.nccl_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("rank worker requires --nccl-id"))?,
    )?;
    let shards = discover_gguf_shards(&args.model)?;
    let device = Device::new_cuda(rank)?;
    unsafe { device.as_cuda_device()?.disable_event_tracking() };
    let _graph_cache_guard = args.cuda_graphs.then(|| {
        device
            .as_cuda_device()
            .expect("CUDA device")
            .enable_cuda_graph_htod_cache()
    });
    let rope = rotary_embedding(&device)?;
    let mut layers = (args.layer..args.layer + args.layers)
        .map(|layer| TensorParallelRankLayer::load(&shards, layer, rank, &device, rope.clone()))
        .collect::<Result<Vec<_>>>()?;
    let weight_bytes = layers
        .iter()
        .map(TensorParallelRankLayer::weight_bytes)
        .sum();
    let mut collective = TensorParallelRankGroup::new(&device, rank, id)?;
    write_response(&WorkerResponse::ready(
        Some(rank),
        weight_bytes,
        Some(collective.nccl_version()),
    ))?;
    worker_loop(|command| match command {
        WorkerCommand::Reset => {
            for layer in &mut layers {
                layer.reset();
            }
            Ok(WorkerResponse::done(0, None))
        }
        WorkerCommand::Forward {
            query_len,
            pos,
            prefill,
            seed,
            broadcast,
            return_output,
        } => {
            let input = deterministic_tensor(&device, query_len, seed, broadcast && rank == 1)?;
            let input = if broadcast {
                collective.broadcast_from_rank0(input)?
            } else {
                input
            };
            let output = rank_forward(&mut layers, input, pos, prefill, &mut collective)?;
            device.synchronize()?;
            let (cache_len, cache_shape) = rank_cache_state(&layers)?;
            let mut response = WorkerResponse::done(cache_len, cache_shape);
            if return_output {
                response.output = Some(output.flatten_all()?.to_vec1::<f32>()?);
            }
            Ok(response)
        }
        WorkerCommand::Benchmark {
            iterations,
            warmup,
            start_pos,
            seed,
            broadcast,
        } => {
            anyhow::ensure!(rank_cache_state(&layers)?.0 == start_pos);
            let input = deterministic_tensor(&device, 1, seed, broadcast && rank == 1)?;
            device.synchronize()?;
            for iteration in 0..warmup {
                let input = if broadcast {
                    collective.broadcast_from_rank0(input.clone())?
                } else {
                    input.clone()
                };
                drop(rank_forward(
                    &mut layers,
                    input,
                    start_pos + iteration,
                    false,
                    &mut collective,
                )?);
            }
            device.synchronize()?;
            let started = Instant::now();
            for iteration in 0..iterations {
                let input = if broadcast {
                    collective.broadcast_from_rank0(input.clone())?
                } else {
                    input.clone()
                };
                drop(rank_forward(
                    &mut layers,
                    input,
                    start_pos + warmup + iteration,
                    false,
                    &mut collective,
                )?);
            }
            device.synchronize()?;
            let (cache_len, cache_shape) = rank_cache_state(&layers)?;
            let mut response = WorkerResponse::done(cache_len, cache_shape);
            response.milliseconds = Some(started.elapsed().as_secs_f64() * 1e3 / iterations as f64);
            Ok(response)
        }
        WorkerCommand::GraphBenchmark {
            iterations,
            warmup,
            pos,
            seed,
            broadcast,
        } => {
            anyhow::ensure!(rank_cache_state(&layers)?.0 == pos);
            let input = deterministic_tensor(&device, 1, seed, broadcast && rank == 1)?;
            device.synchronize()?;
            let stream = device.as_cuda_device()?.cuda_stream();
            stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )?;
            let graph_input = if broadcast {
                collective.broadcast_from_rank0(input.clone())?
            } else {
                input.clone()
            };
            let graph_output = rank_forward(&mut layers, graph_input, pos, false, &mut collective)?;
            let graph = stream
                .end_capture(
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )?
                .ok_or_else(|| anyhow::anyhow!("rank {rank} CUDA graph capture returned no graph"))?;
            graph.upload()?;
            for _ in 0..warmup {
                graph.launch()?;
            }
            device.synchronize()?;
            let started = Instant::now();
            for _ in 0..iterations {
                graph.launch()?;
            }
            device.synchronize()?;
            let (cache_len, cache_shape) = rank_cache_state(&layers)?;
            let mut response = WorkerResponse::done(cache_len, cache_shape);
            response.milliseconds = Some(started.elapsed().as_secs_f64() * 1e3 / iterations as f64);
            response.output = Some(graph_output.flatten_all()?.to_vec1::<f32>()?);
            Ok(response)
        }
        WorkerCommand::Truncate { seq_len } => {
            for layer in &mut layers {
                layer.truncate_cache(seq_len)?;
            }
            let (cache_len, cache_shape) = rank_cache_state(&layers)?;
            Ok(WorkerResponse::done(cache_len, cache_shape))
        }
        WorkerCommand::ClosePeerMapping => {
            collective.close_peer_mapping()?;
            let (cache_len, cache_shape) = rank_cache_state(&layers)?;
            Ok(WorkerResponse::done(cache_len, cache_shape))
        }
        WorkerCommand::FreeLocalPeerBuffer => {
            collective.free_local_peer_buffer()?;
            let (cache_len, cache_shape) = rank_cache_state(&layers)?;
            Ok(WorkerResponse::done(cache_len, cache_shape))
        }
        WorkerCommand::Shutdown => unreachable!("shutdown handled by worker loop"),
    })
}

fn worker_loop(mut handle: impl FnMut(WorkerCommand) -> Result<WorkerResponse>) -> Result<()> {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    while let Some(line) = lines.next() {
        let line = line?;
        let command: WorkerCommand =
            serde_json::from_str(&line).context("decode worker command")?;
        if matches!(command, WorkerCommand::Shutdown) {
            let mut response = WorkerResponse::done(0, None);
            response.status = "shutdown".to_string();
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

fn deterministic_tensor(
    device: &Device,
    query_len: usize,
    seed: u64,
    zero: bool,
) -> Result<Tensor> {
    let count = query_len
        .checked_mul(HIDDEN_SIZE)
        .ok_or_else(|| anyhow::anyhow!("input element count overflows usize"))?;
    let values = if zero {
        vec![0.0; count]
    } else {
        (0..count)
            .map(|index| deterministic_value(index as u64, seed))
            .collect()
    };
    Ok(Tensor::from_vec(
        values,
        (1, query_len, HIDDEN_SIZE),
        device,
    )?)
}

fn deterministic_value(index: u64, seed: u64) -> f32 {
    let mut value = index.wrapping_add(seed).wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let unit = ((value >> 40) as u32) as f32 / (1u32 << 24) as f32;
    unit * 2.0 - 1.0
}

fn compare_values(label: &str, expected: &[f32], actual: &[f32], ceiling: f32) -> Result<()> {
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
    println!("{label}: max_abs={max_abs:.7} mean_abs={mean_abs:.7}");
    anyhow::ensure!(
        max_abs.is_finite() && mean_abs.is_finite(),
        "{label} produced a non-finite difference"
    );
    anyhow::ensure!(
        max_abs <= ceiling,
        "{label} max_abs={max_abs} exceeds {ceiling}"
    );
    Ok(())
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
