import weaviate from 'weaviate-ts-client';
import { config } from './config.js';
import { logger } from './logger.js';

let cached;
export function getWeaviate() {
  if (cached) return cached;
  cached = weaviate.client({
    scheme: 'https',
    host: config.weaviateHost,
    apiKey: new weaviate.ApiKey(config.weaviateApiKey),
  });
  logger.info({ host: config.weaviateHost }, 'Weaviate client ready');
  return cached;
}

const CLASS_MAP = {
  website_structure: 'WebsiteStructure',
  script_example: 'ScriptExample',
  sdk_documentation: 'SDKDocumentation',
};

export function mapClass(dataType = '') {
  return CLASS_MAP[dataType] || 'WebsiteStructure';
}
