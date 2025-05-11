import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { createWorker } from 'tesseract.js';
import { FileText, Upload, Search, MessageSquare } from 'lucide-react';
import { supabase } from './lib/supabase';
import { generateEmbeddings, searchDocuments } from './lib/embeddings';
import { createMistralClient } from './lib/mistral';

function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'search' | 'ask'>('upload');
  const [isProcessing, setIsProcessing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [apiKeys, setApiKeys] = useState({
    mistral: '',
    supabase: ''
  });

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setIsProcessing(true);
    
    try {
      const worker = await createWorker('eng');
      
      for (const file of acceptedFiles) {
        if (file.type === 'application/pdf') {
          // Process PDF with Tesseract
          const result = await worker.recognize(file);
          const text = result.data.text;
          
          // Generate embeddings
          const embedding = await generateEmbeddings(text, apiKeys.mistral);
          
          // Store document and embedding in Supabase
          const { data: document, error: docError } = await supabase
            .from('documents')
            .insert({
              content: text,
              metadata: { filename: file.name }
            })
            .select()
            .single();
            
          if (docError) throw docError;
          
          const { error: embeddingError } = await supabase
            .from('document_embeddings')
            .insert({
              document_id: document.id,
              embedding
            });
            
          if (embeddingError) throw embeddingError;
        }
      }
      
      await worker.terminate();
    } catch (error) {
      console.error('Error processing files:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [apiKeys.mistral]);

  const handleSearch = async () => {
    try {
      const results = await searchDocuments(searchQuery, apiKeys.mistral);
      setSearchResults(results);
    } catch (error) {
      console.error('Error searching documents:', error);
    }
  };

  const handleQuestion = async () => {
    try {
      const client = createMistralClient(apiKeys.mistral);
      const results = await searchDocuments(question, apiKeys.mistral);
      
      const context = results.map(doc => doc.content).join('\n\n');
      const messages = [
        { role: 'system', content: 'You are a helpful assistant. Use the provided context to answer questions.' },
        { role: 'user', content: `Context: ${context}\n\nQuestion: ${question}` }
      ];
      
      const response = await client.chat({
        model: 'mistral-large-latest',
        messages
      });
      
      setAnswer(response.choices[0].message.content);
    } catch (error) {
      console.error('Error processing question:', error);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    }
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold text-gray-900">Document Intelligence Center</h1>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-12 gap-6">
          {/* Sidebar */}
          <div className="col-span-3">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-medium mb-4">Settings</h2>
              
              {/* API Keys */}
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Mistral API Key
                  </label>
                  <input
                    type="password"
                    value={apiKeys.mistral}
                    onChange={(e) => setApiKeys(prev => ({ ...prev, mistral: e.target.value }))}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
              </div>

              {/* Navigation */}
              <nav className="mt-8 space-y-2">
                <button
                  onClick={() => setActiveTab('upload')}
                  className={`w-full flex items-center px-3 py-2 rounded-md ${
                    activeTab === 'upload' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Upload className="mr-2 h-5 w-5" />
                  Upload Documents
                </button>
                
                <button
                  onClick={() => setActiveTab('search')}
                  className={`w-full flex items-center px-3 py-2 rounded-md ${
                    activeTab === 'search' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Search className="mr-2 h-5 w-5" />
                  Search Documents
                </button>
                
                <button
                  onClick={() => setActiveTab('ask')}
                  className={`w-full flex items-center px-3 py-2 rounded-md ${
                    activeTab === 'ask' ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <MessageSquare className="mr-2 h-5 w-5" />
                  Ask Questions
                </button>
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="col-span-9">
            <div className="bg-white rounded-lg shadow">
              {/* Upload Tab */}
              {activeTab === 'upload' && (
                <div className="p-6">
                  <h2 className="text-lg font-medium mb-4">Upload Documents</h2>
                  <div
                    {...getRootProps()}
                    className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-500 transition-colors cursor-pointer"
                  >
                    <input {...getInputProps()} />
                    <FileText className="mx-auto h-12 w-12 text-gray-400" />
                    <p className="mt-2 text-sm text-gray-600">
                      Drag and drop PDF files here, or click to select files
                    </p>
                  </div>
                  {isProcessing && (
                    <div className="mt-4 text-center text-sm text-gray-600">
                      Processing documents...
                    </div>
                  )}
                </div>
              )}

              {/* Search Tab */}
              {activeTab === 'search' && (
                <div className="p-6">
                  <h2 className="text-lg font-medium mb-4">Search Documents</h2>
                  <div className="flex gap-4">
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Enter your search query..."
                      className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                    <button
                      onClick={handleSearch}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    >
                      Search
                    </button>
                  </div>
                  
                  {searchResults.length > 0 && (
                    <div className="mt-6 space-y-4">
                      {searchResults.map((result) => (
                        <div key={result.id} className="p-4 border rounded-lg">
                          <h3 className="font-medium">{result.metadata.filename}</h3>
                          <p className="mt-2 text-sm text-gray-600">{result.content}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Ask Questions Tab */}
              {activeTab === 'ask' && (
                <div className="p-6">
                  <h2 className="text-lg font-medium mb-4">Ask Questions</h2>
                  <div className="space-y-4">
                    <textarea
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="Ask a question about your documents..."
                      rows={4}
                      className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                    <button
                      onClick={handleQuestion}
                      className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    >
                      Submit Question
                    </button>
                    
                    {answer && (
                      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                        <h3 className="font-medium mb-2">Answer:</h3>
                        <p className="text-gray-700">{answer}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;