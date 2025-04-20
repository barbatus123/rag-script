// Lightweight token estimator that works without native tiktoken binaries.
// ----------------------------------------------------------------------
// Heuristic: ~4 characters ≈ 1 token for typical English text. We also
// remove simple HTML tags so they don't inflate the estimate.
// This keeps the code dependency‑free and 100% portable in DigitalOcean
// Functions (or any other constrained runtime).

export function countTokens(text = '') {
    if (typeof text !== 'string') text = String(text || '');

    // naive strip of <> tags to avoid over‑counting markup
    const stripped = text.replace(/<[^>]*?>/g, '');

    return Math.ceil(stripped.length / 4);
}