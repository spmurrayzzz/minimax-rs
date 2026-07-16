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

#[derive(Default)]
pub struct DecodeState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

pub struct MiniMaxTokenizer {
    inner: Tokenizer,
    pub eos: u32,
    pub think_start: u32,
    pub think_end: u32,
    pub tool_start: u32,
    pub tool_end: u32,
}

impl MiniMaxTokenizer {
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
        let vocab: AHashMap<_, _> = tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i as u32))
            .collect();
        let merges: Vec<(String, String)> = merges
            .into_iter()
            .filter_map(|m| {
                let (a, b) = m.split_once(' ')?;
                Some((a.to_owned(), b.to_owned()))
            })
            .collect();
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("[UNK]".into())
            .build()
            .map_err(|e| anyhow::anyhow!("BPE construction failed: {e}"))?;
        let mut inner = Tokenizer::new(bpe);
        let split = Split::new(
            SplitPattern::Regex(MINIMAX_PRETOKENIZER_REGEX.to_owned()),
            SplitDelimiterBehavior::Removed,
            true,
        )
        .map_err(|e| anyhow::anyhow!("MiniMax pre-tokenizer construction failed: {e}"))?;
        let pre_tokenizer = Sequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
        ]);
        inner.with_pre_tokenizer(Some(pre_tokenizer));
        inner.with_decoder(Some(ByteLevelDecoder::default()));
        // Control delimiters are special and should disappear when decoding
        // generated text. Thinking and tool-call tags are ordinary added
        // tokens in the upstream tokenizer and must remain visible to parsers.
        let specials = ["]~!b[", "]~b]", "[e~[", "]!d~[", "[PAD200063]"];
        inner.add_special_tokens(
            &specials
                .iter()
                .map(|s| AddedToken::from(*s, true))
                .collect::<Vec<_>>(),
        );
        let parser_tokens = [
            "<think>",
            "</think>",
            "<minimax:tool_call>",
            "</minimax:tool_call>",
        ];
        inner.add_tokens(
            &parser_tokens
                .iter()
                .map(|s| AddedToken::from(*s, false))
                .collect::<Vec<_>>(),
        );
        let token_id = |token: &str| {
            inner
                .token_to_id(token)
                .with_context(|| format!("missing tokenizer token {token}"))
        };
        let eos = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .context("missing eos")?
            .to_u32()?;
        Ok(Self {
            eos,
            think_start: token_id("<think>")?,
            think_end: token_id("</think>")?,
            tool_start: token_id("<minimax:tool_call>")?,
            tool_end: token_id("</minimax:tool_call>")?,
            inner,
        })
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
        Ok(self
            .inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?)
    }

    pub fn decode_step(&self, state: &mut DecodeState, id: u32) -> Result<Option<String>> {
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
    #[test]
    fn gguf_tokenizer_round_trip() {
        let t = MiniMaxTokenizer::from_gguf(Path::new("/storage/models/minimax-m2.7-gguf/UD-Q4_K_XL/MiniMax-M2.7-UD-Q4_K_XL-00001-of-00004.gguf")).unwrap();
        let ids = t.encode("test").unwrap();
        println!("test ids={ids:?} decoded={:?}", t.decode(&ids).unwrap());
        assert_eq!(t.decode(&ids).unwrap(), "test");
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
    }
}
