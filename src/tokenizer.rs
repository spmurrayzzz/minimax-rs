use ahash::AHashMap;
use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::Content;
use std::{fs::File, path::Path};
use tokenizers::{
    AddedToken, SplitDelimiterBehavior, Tokenizer,
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BPE,
    pre_tokenizers::{
        PreTokenizerWrapper,
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

// tokenizer.ggml.pre = "minimax-m2". GGUF stores the pre-tokenizer name rather
// than its regex, so mirror the canonical MiniMax/GPT-4o split expression used
// by llama.cpp and the original tokenizer.json.
const MINIMAX_PRETOKENIZER_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// llama.cpp promotes these recognized token spellings to end-of-generation in
// addition to the GGUF's explicitly configured EOS/EOT/EOM and FIM IDs. The
// MiniMax vocabulary uses <fim_pad>, <reponame>, and [e~[ (configured EOS).
const LLAMA_EOG_TOKEN_TEXTS: &[&str] = &[
    "<|fim_pad|>",
    "<fim-pad>",
    "<fim_pad>",
    "<PAD>",
    "[PAD]",
    "<|fim_repo|>",
    "<|repo_name|>",
    "<fim-repo>",
    "<REPO>",
    "<reponame>",
    "<|file_sep|>",
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<｜end▁of▁sentence｜>",
];

const CONFIGURED_EOG_METADATA: &[&str] = &[
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.fim_pad_token_id",
    "tokenizer.ggml.fim_rep_token_id",
    "tokenizer.ggml.fim_sep_token_id",
];

#[derive(Default)]
pub struct DecodeState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

pub struct MiniMaxTokenizer {
    inner: Tokenizer,
    eog_ids: Vec<u32>,
    suppressed_ids: Vec<u32>,
    pub think_start: u32,
    pub think_end: u32,
    pub tool_start: u32,
    pub tool_end: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufTokenType {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl From<i32> for GgufTokenType {
    fn from(value: i32) -> Self {
        match value {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Undefined,
        }
    }
}

struct TokenAttributes {
    added_tokens: Vec<AddedToken>,
    suppressed_ids: Vec<u32>,
}

fn token_attributes(tokens: &[String], token_types: &[i32]) -> Result<TokenAttributes> {
    if tokens.len() != token_types.len() {
        anyhow::bail!(
            "tokenizer.ggml.token_type has {} entries for {} tokens",
            token_types.len(),
            tokens.len()
        );
    }

    let mut added_tokens = Vec::new();
    let mut suppressed_ids = Vec::new();
    for (id, (token, token_type)) in tokens.iter().zip(token_types).enumerate() {
        match GgufTokenType::from(*token_type) {
            // llama.cpp recognizes control and unknown spellings only while
            // parsing special tokens, and omits both from ordinary decoding.
            GgufTokenType::Unknown | GgufTokenType::Control => {
                added_tokens.push(AddedToken::from(token.clone(), true).normalized(false));
                suppressed_ids.push(id as u32);
            }
            // User-defined tokens are always recognized and remain visible so
            // the thinking and tool-call parsers can consume their markers.
            GgufTokenType::UserDefined => {
                added_tokens.push(AddedToken::from(token.clone(), false).normalized(false));
            }
            // Undefined and unused pieces are neither recognized in input nor
            // exposed by llama.cpp's ordinary token-to-piece path.
            GgufTokenType::Undefined | GgufTokenType::Unused => {
                suppressed_ids.push(id as u32);
            }
            GgufTokenType::Normal | GgufTokenType::Byte => {}
        }
    }

    Ok(TokenAttributes {
        added_tokens,
        suppressed_ids,
    })
}

fn minimax_pre_tokenizer() -> Result<Sequence> {
    let split = Split::new(
        SplitPattern::Regex(MINIMAX_PRETOKENIZER_REGEX.to_owned()),
        SplitDelimiterBehavior::Removed,
        true,
    )
    .map_err(|e| anyhow::anyhow!("MiniMax pre-tokenizer construction failed: {e}"))?;
    Ok(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        // The MiniMax regex already performs splitting; ByteLevel should only
        // encode bytes here rather than applying its GPT-2 regex again.
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
    ]))
}

impl MiniMaxTokenizer {
    fn from_vocab(
        tokens: Vec<String>,
        merges: Vec<(String, String)>,
        token_types: Vec<i32>,
        eog_ids: Vec<u32>,
    ) -> Result<Self> {
        let attributes = token_attributes(&tokens, &token_types)?;
        let vocab: AHashMap<_, _> = tokens
            .into_iter()
            .enumerate()
            .map(|(i, token)| (token, i as u32))
            .collect();
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("[UNK]".into())
            .build()
            .map_err(|e| anyhow::anyhow!("BPE construction failed: {e}"))?;
        let mut inner = Tokenizer::new(bpe);
        inner.with_pre_tokenizer(Some(minimax_pre_tokenizer()?));
        inner.with_decoder(Some(ByteLevelDecoder::default()));
        inner.add_tokens(&attributes.added_tokens);

        let token_id = |token: &str| {
            inner
                .token_to_id(token)
                .with_context(|| format!("missing tokenizer token {token}"))
        };
        Ok(Self {
            eog_ids,
            suppressed_ids: attributes.suppressed_ids,
            think_start: token_id("<think>")?,
            think_end: token_id("</think>")?,
            tool_start: token_id("<minimax:tool_call>")?,
            tool_end: token_id("</minimax:tool_call>")?,
            inner,
        })
    }

    pub fn from_gguf(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let content = Content::read(&mut file)?;
        let strings = |name: &str| -> Result<Vec<String>> {
            content
                .metadata
                .get(name)
                .context(format!("missing GGUF metadata {name}"))?
                .to_vec()?
                .iter()
                .map(|v| Ok(v.to_string()?.clone()))
                .collect()
        };
        let tokens = strings("tokenizer.ggml.tokens")?;
        let merges = strings("tokenizer.ggml.merges")?;
        let token_types = content
            .metadata
            .get("tokenizer.ggml.token_type")
            .context("missing GGUF metadata tokenizer.ggml.token_type")?
            .to_vec()?
            .iter()
            .map(|value| value.to_i32())
            .collect::<candle_core::Result<Vec<_>>>()?;
        let eos = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .context("missing eos")?
            .to_u32()?;
        let mut eog_ids = vec![eos];
        eog_ids.extend(
            CONFIGURED_EOG_METADATA
                .iter()
                .filter_map(|name| content.metadata.get(*name))
                .map(|value| value.to_u32())
                .collect::<candle_core::Result<Vec<_>>>()?,
        );
        eog_ids.extend(
            tokens
                .iter()
                .enumerate()
                .filter(|(_, token)| LLAMA_EOG_TOKEN_TEXTS.contains(&token.as_str()))
                .map(|(id, _)| id as u32),
        );
        eog_ids.sort_unstable();
        eog_ids.dedup();

        let merges = merges
            .into_iter()
            .filter_map(|merge| {
                let (left, right) = merge.split_once(' ')?;
                Some((left.to_owned(), right.to_owned()))
            })
            .collect();
        Self::from_vocab(tokens, merges, token_types, eog_ids)
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let visible_ids = ids
            .iter()
            .copied()
            .filter(|id| !self.is_suppressed(*id))
            .collect::<Vec<_>>();
        self.inner
            .decode(&visible_ids, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
    }

    fn is_suppressed(&self, id: u32) -> bool {
        self.suppressed_ids.binary_search(&id).is_ok()
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.eog_ids.binary_search(&id).is_ok()
    }

    #[cfg(test)]
    fn eog_ids(&self) -> &[u32] {
        &self.eog_ids
    }

    pub fn decode_step(&self, state: &mut DecodeState, id: u32) -> Result<Option<String>> {
        if self.is_suppressed(id) {
            return Ok(None);
        }
        tokenizers::tokenizer::step_decode_stream(
            &self.inner,
            id,
            true,
            &mut state.ids,
            &mut state.prefix,
            &mut state.prefix_index,
        )
        .map_err(|e| anyhow::anyhow!("stream decode failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};

    #[test]
    fn minimax_pre_tokenizer_keeps_canonical_split_boundaries() -> Result<()> {
        let input = "Hello's 1234!!!\n世界";
        let mut pretokenized: PreTokenizedString = input.into();
        minimax_pre_tokenizer()?
            .pre_tokenize(&mut pretokenized)
            .map_err(|error| anyhow::anyhow!("pre-tokenization failed: {error}"))?;

        let pieces = pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(_, (start, end), _)| &input[start..end])
            .collect::<Vec<_>>();
        assert_eq!(pieces, ["Hello's", " ", "123", "4", "!!!\n", "世界"]);
        Ok(())
    }

    fn token_type_fixture() -> Result<(MiniMaxTokenizer, u32, u32)> {
        let mut tokens = ByteLevel::alphabet()
            .into_iter()
            .map(|character| character.to_string())
            .collect::<Vec<_>>();
        tokens.sort();
        tokens.push("[UNK]".to_owned());
        let mut token_types = vec![1; tokens.len()];
        let mut add_token = |token: &str, token_type: i32| {
            let id = tokens.len() as u32;
            tokens.push(token.to_owned());
            token_types.push(token_type);
            id
        };

        let control_id = add_token("<fim_prefix>", 3);
        let think_start = add_token("<think>", 4);
        let think_end = add_token("</think>", 4);
        let tool_start = add_token("<minimax:tool_call>", 4);
        let tool_end = add_token("</minimax:tool_call>", 4);
        let unused_id = add_token("[PAD_UNUSED]", 5);
        let tokenizer = MiniMaxTokenizer::from_vocab(tokens, vec![], token_types, vec![])?;
        assert_eq!(tokenizer.think_start, think_start);
        assert_eq!(tokenizer.think_end, think_end);
        assert_eq!(tokenizer.tool_start, tool_start);
        assert_eq!(tokenizer.tool_end, tool_end);
        Ok((tokenizer, control_id, unused_id))
    }

    #[test]
    fn gguf_token_types_drive_encode_and_decode() -> Result<()> {
        let (tokenizer, control_id, unused_id) = token_type_fixture()?;

        assert_eq!(tokenizer.encode("<fim_prefix>")?, [control_id]);
        assert_eq!(tokenizer.decode(&[control_id])?, "");
        assert_eq!(tokenizer.encode("<think>")?, [tokenizer.think_start]);
        assert_eq!(
            tokenizer.decode(&[
                tokenizer.think_start,
                tokenizer.think_end,
                tokenizer.tool_start,
                tokenizer.tool_end,
            ])?,
            "<think></think><minimax:tool_call></minimax:tool_call>"
        );
        assert_ne!(tokenizer.encode("[PAD_UNUSED]")?, [unused_id]);
        assert_eq!(tokenizer.decode(&[unused_id])?, "");

        let mut state = DecodeState::default();
        assert_eq!(tokenizer.decode_step(&mut state, control_id)?, None);
        assert_eq!(
            tokenizer.decode_step(&mut state, tokenizer.think_start)?,
            Some("<think>".to_owned())
        );
        assert_eq!(tokenizer.decode_step(&mut state, unused_id)?, None);
        assert_eq!(
            tokenizer.decode_step(&mut state, tokenizer.tool_start)?,
            Some("<minimax:tool_call>".to_owned())
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MiniMax GGUF weights; set MINIMAX_MODEL_DIR"]
    fn gguf_tokenizer_round_trip() -> Result<()> {
        let model_dir = std::env::var_os(minimax::model_files::MODEL_DIR_ENV)
            .map(std::path::PathBuf::from)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} must point to the directory containing the GGUF shards",
                    minimax::model_files::MODEL_DIR_ENV
                )
            })?;
        let shards = minimax::model_files::discover_gguf_shards(&model_dir)?;
        let t = MiniMaxTokenizer::from_gguf(&shards[0])?;
        let ids = t.encode("test").unwrap();
        println!("test ids={ids:?} decoded={:?}", t.decode(&ids).unwrap());
        assert_eq!(t.decode(&ids).unwrap(), "test");
        let expected_suppressed = (200000..=200049).chain(200054..=200063).collect::<Vec<_>>();
        assert_eq!(t.suppressed_ids, expected_suppressed);
        for id in 200000..=200049 {
            let token = t.inner.id_to_token(id).context("missing control token")?;
            assert_eq!(t.encode(&token)?, [id], "control token {id}: {token}");
            assert_eq!(t.decode(&[id])?, "", "control token {id}: {token}");
        }
        for id in 200050..=200053 {
            let token = t
                .inner
                .id_to_token(id)
                .context("missing user-defined token")?;
            assert_eq!(t.encode(&token)?, [id], "user-defined token {id}: {token}");
            assert_eq!(
                t.decode(&[id])?,
                token,
                "user-defined token {id} must remain visible"
            );
        }
        for id in 200054..=200063 {
            let token = t.inner.id_to_token(id).context("missing unused token")?;
            assert_ne!(t.encode(&token)?, [id], "unused token {id}: {token}");
            assert_eq!(t.decode(&[id])?, "", "unused token {id}: {token}");
        }
        assert_eq!(t.encode("[PAD200063]")?, [91, 115375, 866, 51821, 93]);
        assert_eq!(t.encode("]~!b[").unwrap(), vec![200034]);
        assert_eq!(t.encode("[e~[").unwrap(), vec![200020]);
        assert_eq!(t.encode("]~b]system").unwrap(), vec![200019, 28463]);
        assert_eq!(t.encode("]~b]user").unwrap(), vec![200019, 3995]);
        assert_eq!(t.encode("]~b]ai").unwrap(), vec![200019, 1361]);
        assert_eq!(t.encode("-M").unwrap(), vec![5145]);
        assert_eq!(t.think_start, 200050);
        assert_eq!(t.think_end, 200051);
        assert_eq!(t.tool_start, 200052);
        assert_eq!(t.tool_end, 200053);
        assert_eq!(t.eog_ids(), &[200004, 200005, 200020]);
        for id in [200004, 200005, 200020] {
            assert!(t.is_eog(id), "expected token {id} to be EOG");
        }
        assert!(!t.is_eog(200019));
        assert_eq!(
            t.decode(&[t.think_end, t.tool_start, t.tool_end]).unwrap(),
            "</think><minimax:tool_call></minimax:tool_call>"
        );
        let prompt = "]~!b[]~b]system\nYou are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.[e~[\n]~b]user\ntest[e~[\n]~b]ai\n<think>\n";
        assert_eq!(
            t.encode(prompt).unwrap(),
            vec![
                200034, 200019, 28463, 10, 2985, 457, 258, 12473, 23413, 46, 5324, 1925, 355,
                35353, 12973, 5145, 50, 46, 55, 306, 355, 6904, 531, 35353, 12973, 46, 200020, 10,
                200019, 3995, 10, 4500, 200020, 10, 200019, 1361, 10, 200050, 10,
            ]
        );
        let golden = "The user says \"test\". This is a simple test message. The assistant should respond appropriately. There's no policy violation.";
        assert_eq!(
            t.encode(golden).unwrap(),
            vec![
                758, 3100, 3700, 494, 4500, 4969, 1204, 355, 258, 4160, 1618, 4838, 46, 517, 23413,
                1352, 7623, 36238, 46, 25209, 687, 5177, 23077, 46,
            ]
        );
        Ok(())
    }
}
