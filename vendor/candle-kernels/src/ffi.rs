use core::ffi::c_void;
#[allow(dead_code)]
#[allow(improper_ctypes)]
extern "C" {
    // for unquntized models
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void,      // device pointer [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_gemm_gguf(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    pub fn minimax_moe_decode_topk(
        logits: *const f32,
        bias: *const f32,
        topk_ids: *mut u32,
        topk_weights: *mut f32,
        expert_ids: *mut u32,
        route_ids: *mut u32,
        stream: i64,
    );

    pub fn moe_gemm_gguf_decode_silu(
        input: *const f32,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        down_weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut f32,
        num_experts: i32,
        topk: i32,
        hidden_size: i32,
        intermediate_size: i32,
        gate_up_gguf_dtype: i32,
        down_gguf_dtype: i32,
        stream: i64,
    );

    pub fn moe_gemm_gguf_prefill(
        input: *const c_void, // input [size_m, size_k]
        weights: *const u8,   // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,   //must be host ptr
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
        gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    pub fn moe_gemm_mmq_gguf(
        input: *const f32,      // input [source_rows, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut f32,         // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32,
        stream: i64,
    );

    pub fn moe_sort_routes(
        expert_ids_unsorted: *const u32,
        expert_ids_sorted: *mut u32,
        route_ids_sorted: *mut u32,
        size: i32,
        end_bit: i32,
        stream: i64,
    );

    // ============== Dense GGUF MMVQ launchers (from mmvq_gguf.cu) ==============

    // BF16 output launchers
    pub fn launch_mmvq_gguf_q4_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    // F32 output launchers
    pub fn launch_mmvq_gguf_q4_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    pub fn launch_mmvq_gguf_q4_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    // Quantize launchers (activation → Q8_1)
    pub fn launch_mmvq_gguf_quantize_q8_1_bf16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_quantize_q8_1_f16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_quantize_q8_1_f32(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_qkv_q8_0_f32(
        x: *const c_void,
        vy: *mut c_void,
        vq: *const c_void,
        vk: *const c_void,
        vv: *const c_void,
        dst: *mut c_void,
        kx: i32,
        kx_padded: i32,
        q_rows: i32,
        k_rows: i32,
        v_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_qkv_q8_0_f32_rmsnorm_f16(
        x: *const c_void,
        vy: *mut c_void,
        vq: *const c_void,
        vk: *const c_void,
        vv: *const c_void,
        projection_dst: *mut c_void,
        q_alpha: *const c_void,
        k_alpha: *const c_void,
        dst: *mut c_void,
        kx: i32,
        kx_padded: i32,
        q_rows: i32,
        k_rows: i32,
        v_rows: i32,
        eps: f32,
        stream: *mut c_void,
    );
    pub fn launch_tp2_peer_all_reduce_f32(
        local: *const c_void,
        peer: *const c_void,
        residual: *const c_void,
        norm_weight: *const c_void,
        output: *mut c_void,
        normalized_output: *mut c_void,
        self_signal: *mut c_void,
        peer_signal: *mut c_void,
        epoch: u32,
        count: i32,
        norm_eps: f32,
        stream: *mut c_void,
    );
    pub fn launch_tp2_peer_broadcast_f32(
        source: *const c_void,
        output: *mut c_void,
        self_signal: *mut c_void,
        peer_signal: *mut c_void,
        epoch: u32,
        rank: i32,
        count: i32,
        stream: *mut c_void,
    );
    pub fn launch_qkv_tp_sumsq_f32(
        q: *const c_void,
        k: *const c_void,
        stats: *mut c_void,
        q_rows: i32,
        k_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_qkv_tp_rmsnorm_rope_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        global_stats: *const c_void,
        q_alpha: *const c_void,
        k_alpha: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        dst: *mut c_void,
        q_rows: i32,
        k_rows: i32,
        v_rows: i32,
        q_full_rows: i32,
        k_full_rows: i32,
        head_dim: i32,
        rope_dim: i32,
        position: i32,
        eps: f32,
        stream: *mut c_void,
    );
    pub fn launch_partial_rope_f16(
        q: *const c_void,
        k: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        dst: *mut c_void,
        q_rows: i32,
        k_rows: i32,
        head_dim: i32,
        rope_dim: i32,
        position: i32,
        stream: *mut c_void,
    );
    pub fn launch_gqa_decode_mma_f16_128(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        partials: *mut c_void,
        output: *mut c_void,
        seq_len: i32,
        q_head_stride: i32,
        k_head_stride: i32,
        v_head_stride: i32,
        num_splits: i32,
        query_heads: i32,
        kv_heads: i32,
        scale: f32,
        device_ordinal: i32,
        stream: *mut c_void,
    );
    pub fn launch_gqa_f16_128(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        partials: *mut c_void,
        output: *mut c_void,
        query_len: i32,
        past_len: i32,
        q_head_stride: i32,
        q_seq_stride: i32,
        k_head_stride: i32,
        v_head_stride: i32,
        num_splits: i32,
        query_heads: i32,
        kv_heads: i32,
        scale: f32,
        stream: *mut c_void,
    );

    // ============== Dense GGUF MMQ launchers (from mmq_gguf/) ==============

    // MMQ quantize launchers (f32 -> block_q8_1_mmq, 3 scale layouts)
    pub fn launch_mmq_quantize_q8_1_D4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_DS4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_D2S6(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );

    // MMQ matmul launchers (one per quant type)
    pub fn launch_mmq_gguf_q4_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q8_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q2_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q3_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q6_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
}
