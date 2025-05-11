import { supabase } from './supabase';
import { createMistralClient } from './mistral';

export async function generateEmbeddings(text: string, apiKey: string) {
  const client = createMistralClient(apiKey);
  const response = await client.embeddings({
    model: "mistral-embed",
    input: text
  });
  return response.data[0].embedding;
}

export async function searchDocuments(query: string, apiKey: string) {
  const queryEmbedding = await generateEmbeddings(query, apiKey);

  const { data: documents, error } = await supabase.rpc('match_documents', {
    query_embedding: queryEmbedding,
    match_threshold: 0.7,
    match_count: 10
  });

  if (error) throw error;
  return documents;
}