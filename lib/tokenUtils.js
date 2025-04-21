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

export function countTokens(text) {
  const tokens = enc.encode(text);
  return tokens.length;
}
