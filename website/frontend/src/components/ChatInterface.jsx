import React, { useState, useEffect } from 'react';
import { Send, Trash2, Edit2, ChevronRight, ChevronLeft, BarChart2, Check, Settings} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = {
  RUN_INFERENCE: "https://eduport-tech--emelinlabs-runner-run-inference.modal.run",
  GET_RESULT: "https://eduport-tech--emelinlabs-runner-get-result.modal.run",
  STREAM_ORIGINAL: "https://eduport-tech--emelinlabs-runner-stream-original.modal.run",
  STREAM_QUANTIZED: "https://eduport-tech--emelinlabs-runner-stream-quantized.modal.run"
};

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [editingMessageId, setEditingMessageId] = useState(null);
  const [editingText, setEditingText] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState('Qwen/Qwen2.5-1.5B');
  const [quantizationBits, setQuantizationBits] = useState(6);
  const [quantizationType, setQuantizationType] = useState('WASQ-OPT');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [responseType, setResponseType] = useState('both');

  const makeApiRequest = async (url, options) => {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${error.message}`);
      throw error;
    }
  };

<<<<<<< HEAD
  // Poll for results function
  const pollForResults = async (requestId, maxAttempts = 100, interval = 6000) => {
=======
  const pollForResults = async (requestId, maxAttempts = 20, interval = 6000) => {
>>>>>>> 13e8bad77676836a378020fd3af242b37b8a4b31
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await makeApiRequest(
          `${API_BASE_URL.GET_RESULT}?request_id=${requestId}`,
          { method: 'GET' }
        );
        if (response?.original) return response;
      } catch (error) {
        console.warn(`Polling attempt ${attempt + 1} failed:`, error);
      }
      await new Promise(resolve => setTimeout(resolve, interval));
    }
    throw new Error('Timeout waiting for response');
  };

  const handleStreaming = (requestId) => {
    setStreaming(true);
    const eventSources = [];
    
    if (responseType === 'both' || responseType === 'original') {
      const originalEventSource = new EventSource(`${API_BASE_URL.STREAM_ORIGINAL}?request_id=${requestId}`);
      eventSources.push(originalEventSource);
      let originalText = '';
      originalEventSource.onmessage = (event) => {
        originalText += event.data + ' ';
        setMessages(prev => updateMessageText(prev, 'original', originalText));
      };
    }

    if (responseType === 'both' || responseType === 'quantized') {
      const quantizedEventSource = new EventSource(`${API_BASE_URL.STREAM_QUANTIZED}?request_id=${requestId}`);
      eventSources.push(quantizedEventSource);
      let quantizedText = '';
      quantizedEventSource.onmessage = (event) => {
        quantizedText += event.data + ' ';
        setMessages(prev => updateMessageText(prev, 'quantized', quantizedText));
      };
    }

    const errorHandler = () => {
      eventSources.forEach(es => es.close());
      setStreaming(false);
    };

    eventSources.forEach(es => es.onerror = errorHandler);
  };

  const updateMessageText = (prevMessages, type, text) => {
    const lastMessage = prevMessages[prevMessages.length - 1];
    return lastMessage?.sender === 'bot' ? [
      ...prevMessages.slice(0, -1),
      {
        ...lastMessage,
        metrics: {
          ...lastMessage.metrics,
          [type]: { ...lastMessage.metrics[type], text: text.trim() }
        }
      }
    ] : prevMessages;
  };

  const runInference = async (inputText) => {
    try {
      setLoading(true);
      const requestData = {
        model_name: selectedModel,
        quantization_bits: quantizationBits,
        quantization_type: quantizationType,
        input_text: inputText,
      };

      const postResponse = await makeApiRequest(API_BASE_URL.RUN_INFERENCE, {
        method: 'POST',
        body: JSON.stringify(requestData),
      });

      if (!postResponse?.request_id) throw new Error('Invalid response from server');
      
      handleStreaming(postResponse.request_id);
      const results = await pollForResults(postResponse.request_id);

      return {
        text: responseType === 'quantized' ? results.quantized.text : results.original.text,
        metrics: {
          original: responseType !== 'quantized' ? {
            text: results.original.text,
            inferenceTime: results.original.inference_time,
            memoryUsage: results.original.memory_usage,
            perplexity: results.original.perplexity
          } : null,
          quantized: responseType !== 'original' ? {
            text: results.quantized.text,
            inferenceTime: results.quantized.inference_time,
            memoryUsage: results.quantized.memory_usage,
            perplexity: results.quantized.perplexity
          } : null,
          comparison: responseType === 'both' ? results.comparison : null
        }
      };
    } catch (error) {
      console.error('Error in runInference:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
    setEditingMessageId(null);
    setEditingText('');
  };

  const handleEdit = (messageId) => {
    const messageToEdit = messages.find(msg => msg.id === messageId);
    if (messageToEdit) {
      setEditingMessageId(messageId);
      setEditingText(messageToEdit.text);
    }
  };

  const handleSend = async () => {
    if (!inputMessage.trim() || loading) return;
    const userMessage = { 
      id: Date.now(), 
      text: inputMessage, 
      sender: 'user',
      responseType: null // User messages don't need response type
    };
    
    try {
      setMessages(prev => [...prev, userMessage]);
      setInputMessage('');
      const results = await runInference(inputMessage);
      
      const botMessage = {
        id: Date.now() + 1,
        text: results.text,
        sender: 'bot',
        metrics: results.metrics,
        responseType: responseType // Store current response type with message
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: `Error: ${error.message}. Please try again later.`,
        sender: 'bot',
        responseType: responseType
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleSaveEdit = async (messageId) => {
    try {
      const messageIndex = messages.findIndex(msg => msg.id === messageId);
      const editedMessage = messages[messageIndex];
      if (messageIndex === -1 || editedMessage.sender !== 'user') return;
      
      const updatedMessages = messages.slice(0, messageIndex + 1);
      updatedMessages[messageIndex] = { ...editedMessage, text: editingText };
      setMessages(updatedMessages);
      setEditingMessageId(null);
      setEditingText('');
      
      setLoading(true);
      const results = await runInference(editingText);

      const botMessage = {
        id: Date.now(),
        text: results.text,
        sender: 'bot',
        metrics: results.metrics,
        responseType: responseType // Store current response type with new message
      };

      setMessages([...updatedMessages, botMessage]);
    } catch (error) {
      console.error('Error in handleSaveEdit:', error);
      const errorMessage = {
        id: Date.now(),
        text: `Error: ${error.message}. Please try again later.`,
        sender: 'bot',
        responseType: responseType
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const MessageBubble = ({ message, children }) => (
    <div className={`p-4 rounded-2xl max-w-3xl ${
      message.sender === 'user' 
        ? 'bg-[#27272A] text-white ml-auto'
        : 'bg-[#27272A] text-gray-100'
    }`}>
      {children}
    </div>
  );

  const MetricBadge = ({ label, value }) => (
    <div className="flex items-center gap-2 bg-[#18181B] px-3 py-1.5 rounded-full">
      <span className="text-xs text-[#A1A1AA]">{label}</span>
      <span className="text-sm font-medium text-[#E4E4E7]">{value}</span>
    </div>
  );

  const ModelResponseSection = ({ title, text, metrics = {}, type }) => {
    const [showMetrics, setShowMetrics] = useState(false);
  
    return (
      <div className="space-y-4 relative group">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`h-2 w-2 rounded-full ${type === 'original' ? 'bg-green-500' : 'bg-purple-500'}`} />
            <h3 className="font-semibold text-gray-200">{title}</h3>
          </div>
          <button 
            onMouseEnter={() => setShowMetrics(true)}
            onMouseLeave={() => setShowMetrics(false)}
            className="p-1.5 text-[#A1A1AA] hover:text-white rounded-lg hover:bg-[#3F3F46] transition-colors"
          >
            <BarChart2 size={18} />
          </button>
        </div>
  
        <div className="prose prose-invert max-w-none text-gray-300">
          <ReactMarkdown>{text}</ReactMarkdown>
        </div>
  
        {/* Metrics Panel */}
        {showMetrics && (
          <div 
            className="absolute top-8 right-0 z-10 p-4 bg-[#18181B] rounded-xl shadow-xl border border-[#3F3F46] w-64"
            onMouseEnter={() => setShowMetrics(true)}
            onMouseLeave={() => setShowMetrics(false)}
          >
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-[#A1A1AA]">Inference Time</span>
                <span className="text-sm font-medium text-[#E4E4E7]">
                  {metrics?.inferenceTime?.toFixed(2) || '0.00'}s
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-[#A1A1AA]">Memory Usage</span>
                <span className="text-sm font-medium text-[#E4E4E7]">
                  {metrics?.memoryUsage?.toFixed(2) || '0.00'}MB
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-[#A1A1AA]">Perplexity</span>
                <span className="text-sm font-medium text-[#E4E4E7]">
                  {metrics?.perplexity?.toFixed(2) || '0.00'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };


 
  const MessageComparison = ({ message }) => {
    const [showStats, setShowStats] = useState(false);

    if (message.sender === 'user') {
      return (
        <div className="group relative mb-6">
          {editingMessageId === message.id ? (
            <div className="flex gap-2">
              <input
                type="text"
                value={editingText}
                onChange={(e) => setEditingText(e.target.value)}
                className="flex-1 bg-zinc-900 text-white rounded-xl p-3 border border-zinc-700 focus:ring-2 focus:ring-blue-500"
                onKeyDown={(e) => e.key === 'Enter' && handleSaveEdit(message.id)}
                autoFocus
              />
              <button
                onClick={() => handleSaveEdit(message.id)}
                className="p-2 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors"
              >
                <Check size={20} />
              </button>
            </div>
          ) : (
            <div className="relative text-left">
            <MessageBubble message={message}>
              {message.text}
            </MessageBubble>
            <button
              onClick={() => handleEdit(message.id)}
              className="absolute -right-8 top-1/2 -translate-y-1/2 p-1.5 text-[#A1A1AA] hover:text-white transition-colors"
            >
              <Edit2 size={18} />
            </button>
          </div>
          )}
        </div>
      );
    }

    return (
      <div className="mb-6 space-y-6">
        <div className="relative">
          <div className={`grid gap-6 ${message.responseType === 'both' ? 'grid-cols-2' : 'grid-cols-1'}`}>
            {message.responseType !== 'quantized' && (
              <ModelResponseSection
                title="Original Model"
                text={message.metrics?.original?.text || message.text}
                metrics={message.metrics?.original}
                type="original"
              />
            )}

            {message.responseType !== 'original' && (
              <ModelResponseSection
                title="Quantized Model"
                text={message.metrics?.quantized?.text || message.text}
                metrics={message.metrics?.quantized}
                type="quantized"
              />
            )}
          </div>

          {message.metrics?.comparison && message.responseType === 'both' && (
            <div className="absolute top-0 right-0">
              <div className="relative">
                <button
                  onMouseEnter={() => setShowStats(true)}
                  onMouseLeave={() => setShowStats(false)}
                  className="p-2 bg-zinc-900 hover:bg-zinc-700 rounded-xl text-zinc-400 hover:text-white transition-colors"
                >
                  <BarChart2 size={20} />
                </button>

                {showStats && (
                  <div className="absolute right-0 mt-2 w-64 p-4 bg-zinc-900 rounded-xl shadow-xl border border-zinc-700 z-10">
                    <div className="space-y-3 text-sm">
                      <div className="flex justify-between items-center">
                        <span className="text-zinc-400">Speed:</span>
                        <span className="text-green-500 font-medium">
                          +{message.metrics.comparison.speed_diff.toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-zinc-400">Memory:</span>
                        <span className="text-purple-500 font-medium">
                          -{message.metrics.comparison.memory_savings.toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-zinc-400">Quality:</span>
                        <span className="text-zinc-200 font-medium">
                          {message.metrics.comparison.quality_diff.toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };
  return (
    <div className="h-screen flex bg-zinc-950">
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} bg-zinc-900 border-r border-zinc-700 transition-all duration-300 overflow-hidden`}>
        <div className="p-6 space-y-8">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-200">Model Settings</h2>
            <Settings className="text-zinc-400" size={20} />
          </div>
          
          <div className="space-y-6">
            <div className="space-y-3">
              <label className="text-sm font-medium text-gray-300">Model Selection</label>
              <select 
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-xl p-3 text-gray-200 focus:ring-2 focus:ring-blue-500"
              >
                <option value="Qwen/Qwen2.5-0.5B">Qwen/Qwen2.5-0.5B</option>
                <option value="Qwen/Qwen2.5-1.5B">Qwen/Qwen2.5-1.5B</option>
                <option value="meta-llama/Llama-3.2-1B">meta-llama/Llama-3.2-1B</option>
                <option value="meta-llama/Llama-3.2-3B">meta-llama/Llama-3.2-3B</option>
                <option value="meta-llama/Llama-3.1-8B">meta-llama/Llama-3.1-8B</option>
              </select>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium text-gray-300">Quantization Bits</label>
                <span className="text-sm text-blue-500">{quantizationBits} bits</span>
              </div>
              <input
                type="range"
                min="2"
                max="20"
                value={quantizationBits}
                onChange={(e) => setQuantizationBits(parseInt(e.target.value))}
                className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium text-gray-300">Quantization Type</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setQuantizationType('WASQ-OPT')}
                  className={`p-2 rounded-lg text-sm ${
                    quantizationType === 'WASQ-OPT'
                      ? 'bg-blue-500 text-white'
                      : 'bg-zinc-800 text-gray-300 hover:bg-zinc-700'
                  }`}
                >
                  WASQ-OPT
                </button>
                <button
                  onClick={() => setQuantizationType('WASQ-LTH')}
                  className={`p-2 rounded-lg text-sm ${
                    quantizationType === 'WASQ-LTH'
                      ? 'bg-blue-500 text-white'
                      : 'bg-zinc-800 text-gray-300 hover:bg-zinc-700'
                  }`}
                >
                  WASQ-LTH
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col relative">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="absolute left-4 top-4 z-10 p-2 bg-zinc-900 text-zinc-400 hover:text-white rounded-xl hover:bg-zinc-700 transition-colors"
        >
          {sidebarOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
        </button>

        <div className="flex-1 overflow-y-auto p-6 pt-16">
          <div className="max-w-3xl mx-auto space-y-12">
            {messages.map((message) => (
              <MessageComparison key={message.id} message={message} />
            ))}
            {loading && (
              <div className="flex justify-center">
<<<<<<< HEAD
                <div className="bg-[#1D1D1F] rounded-lg p-4 text-white">
                  Processing request... This may take up to 10 minutes.
=======
                <div className="animate-pulse bg-zinc-900 rounded-xl p-4 text-gray-400">
                  Generating response...
>>>>>>> 13e8bad77676836a378020fd3af242b37b8a4b31
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-zinc-700 bg-zinc-900 p-6">
          <div className="max-w-3xl mx-auto flex gap-3">
            <div className="flex gap-1 bg-zinc-800 p-1 rounded-xl">
              {[ 'original', 'quantized', 'both'].map((type) => (
                <button
                  key={type}
                  onClick={() => setResponseType(type)}
                  className={`px-4 py-2 rounded-lg text-sm capitalize ${
                    responseType === type
                      ? 'bg-blue-500 text-white'
                      : 'text-gray-400 hover:bg-zinc-700'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
            
            <div className="flex-1 relative">
              <input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                placeholder="Message SuperFloat"
                className="w-full pr-14 pl-4 py-3 bg-zinc-800 text-gray-200 rounded-xl border border-zinc-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={loading}
                className="absolute right-2 top-2 p-2 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={20} />
              </button>
            </div>
            <button
              onClick={handleClear}
              className="p-3 text-zinc-400 hover:text-red-500 hover:bg-zinc-700 rounded-xl transition-colors"
            >
              <Trash2 size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// API interaction functions
const makeApiRequest = async (url, options) => {
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error(`API request failed: ${error.message}`);
    throw error;
  }
};

const pollForResults = async (requestId, maxAttempts = 20, interval = 6000) => {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const response = await makeApiRequest(
        `${API_BASE_URL.GET_RESULT}?request_id=${requestId}`,
        { method: 'GET' }
      );
      if (response?.original) return response;
    } catch (error) {
      console.warn(`Polling attempt ${attempt + 1} failed:`, error);
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }
  throw new Error('Timeout waiting for response');
};

const handleStreaming = (requestId, responseType, updateMessageText) => {
  const eventSources = [];
  
  if (responseType === 'both' || responseType === 'original') {
    const originalEventSource = new EventSource(`${API_BASE_URL.STREAM_ORIGINAL}?request_id=${requestId}`);
    eventSources.push(originalEventSource);
    let originalText = '';
    originalEventSource.onmessage = (event) => {
      originalText += event.data + ' ';
      updateMessageText('original', originalText.trim());
    };
  }

  if (responseType === 'both' || responseType === 'quantized') {
    const quantizedEventSource = new EventSource(`${API_BASE_URL.STREAM_QUANTIZED}?request_id=${requestId}`);
    eventSources.push(quantizedEventSource);
    let quantizedText = '';
    quantizedEventSource.onmessage = (event) => {
      quantizedText += event.data + ' ';
      updateMessageText('quantized', quantizedText.trim());
    };
  }

  return () => eventSources.forEach(es => es.close());
};

const runInference = async (inputText, selectedModel, quantizationBits, quantizationType, responseType) => {
  try {
    const requestData = {
      model_name: selectedModel,
      quantization_bits: quantizationBits,
      quantization_type: quantizationType,
      input_text: inputText,
    };

    const postResponse = await makeApiRequest(API_BASE_URL.RUN_INFERENCE, {
      method: 'POST',
      body: JSON.stringify(requestData),
    });

    if (!postResponse?.request_id) throw new Error('Invalid response from server');
    
    const results = await pollForResults(postResponse.request_id);

    return {
      text: responseType === 'quantized' ? results.quantized.text : results.original.text,
      metrics: {
        original: responseType !== 'quantized' ? {
          text: results.original.text,
          inferenceTime: results.original.inference_time,
          memoryUsage: results.original.memory_usage,
          perplexity: results.original.perplexity
        } : null,
        quantized: responseType !== 'original' ? {
          text: results.quantized.text,
          inferenceTime: results.quantized.inference_time,
          memoryUsage: results.quantized.memory_usage,
          perplexity: results.quantized.perplexity
        } : null,
        comparison: responseType === 'both' ? results.comparison : null
      }
    };
  } catch (error) {
    console.error('Error in runInference:', error);
    throw error;
  }
};

export default ChatInterface;
