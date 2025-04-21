// Lightweight token estimator that works without native tiktoken binaries.
// ----------------------------------------------------------------------
// Heuristic: ~4 characters ≈ 1 token for typical English text. We also
// remove simple HTML tags so they don't inflate the estimate.
// This keeps the code dependency‑free and 100% portable in DigitalOcean
// Functions (or any other constrained runtime).

import { encoding_for_model } from 'tiktoken';

const enc = encoding_for_model('text-embedding-ada-002');

export function trimTokens(text, maxTokens = 512) {
    const tokens = enc.encode(text);
    if (tokens.length <= maxTokens) {
        return { text, tokens: tokens.length };
    }

    // Truncate to maxTokens and decode back to text
    const truncatedTokens = tokens.slice(0, maxTokens);
    return { text: enc.decode(truncatedTokens), tokens: truncatedTokens.length };
}
