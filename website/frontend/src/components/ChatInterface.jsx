import React, { useState, useEffect } from 'react';
import { Send, Trash2, Edit2, ChevronRight, ChevronLeft, BarChart2, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = {
  RUN_INFERENCE: "https://eduport-tech--emelinlabs-runner-run-inference-dev.modal.run",
  GET_RESULT: "https://eduport-tech--emelinlabs-runner-get-result-dev.modal.run",
  STREAM_ORIGINAL: "https://eduport-tech--emelinlabs-runner-stream-original-dev.modal.run",
  STREAM_QUANTIZED: "https://eduport-tech--emelinlabs-runner-stream-quantized-dev.modal.run"
};

const ChatInterface = () => {
  // State management
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

  // API request helper function
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

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${error.message}`);
      throw new Error(`API request failed: ${error.message}`);
    }
  };

  // Poll for results function
  const pollForResults = async (requestId, maxAttempts = 20, interval = 6000) => {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await makeApiRequest(
          `${API_BASE_URL.GET_RESULT}?request_id=${requestId}`,
          { method: 'GET' }
        );
        
        if (response && response.original) {
          return response;
        }
      } catch (error) {
        console.warn(`Polling attempt ${attempt + 1} failed:`, error);
      }
      
      await new Promise(resolve => setTimeout(resolve, interval));
    }
    throw new Error('Timeout waiting for response');
  };

  // Handle streaming responses
  const handleStreaming = (requestId) => {
    setStreaming(true);

    const originalEventSource = new EventSource(`${API_BASE_URL.STREAM_ORIGINAL}?request_id=${requestId}`);
    const quantizedEventSource = new EventSource(`${API_BASE_URL.STREAM_QUANTIZED}?request_id=${requestId}`);

    let originalText = '';
    let quantizedText = '';

    originalEventSource.onmessage = (event) => {
      originalText += event.data + ' ';
      setMessages(prevMessages => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        if (lastMessage && lastMessage.sender === 'bot') {
          return [
            ...prevMessages.slice(0, -1),
            {
              ...lastMessage,
              metrics: {
                ...lastMessage.metrics,
                original: {
                  ...lastMessage.metrics.original,
                  text: originalText.trim(),
                },
              },
            },
          ];
        }
        return prevMessages;
      });
    };

    quantizedEventSource.onmessage = (event) => {
      quantizedText += event.data + ' ';
      setMessages(prevMessages => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        if (lastMessage && lastMessage.sender === 'bot') {
          return [
            ...prevMessages.slice(0, -1),
            {
              ...lastMessage,
              metrics: {
                ...lastMessage.metrics,
                quantized: {
                  ...lastMessage.metrics.quantized,
                  text: quantizedText.trim(),
                },
              },
            },
          ];
        }
        return prevMessages;
      });
    };

    originalEventSource.onerror = () => {
      originalEventSource.close();
      setStreaming(false);
    };

    quantizedEventSource.onerror = () => {
      quantizedEventSource.close();
      setStreaming(false);
    };
  };

  // Run inference function
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

      if (!postResponse || !postResponse.request_id) {
        throw new Error('Invalid response from server');
      }

      handleStreaming(postResponse.request_id);

      const results = await pollForResults(postResponse.request_id);

      return {
        text: results.original.text,
        metrics: {
          original: {
            text: results.original.text,
            inferenceTime: results.original.inference_time,
            memoryUsage: results.original.memory_usage,
            perplexity: results.original.perplexity
          },
          quantized: {
            text: results.quantized.text,
            inferenceTime: results.quantized.inference_time,
            memoryUsage: results.quantized.memory_usage,
            perplexity: results.quantized.perplexity
          },
          comparison: results.comparison
        }
      };
    } catch (error) {
      console.error('Error in runInference:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  // Handle sending messages
  const handleSend = async () => {
    if (!inputMessage.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
    };

    try {
      setMessages(prev => [...prev, userMessage]);
      setInputMessage('');

      const results = await runInference(inputMessage);
      
      const botMessage = {
        id: Date.now() + 1,
        text: results.text,
        sender: 'bot',
        metrics: results.metrics
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: `Error: ${error.message}. Please try again later.`,
        sender: 'bot',
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  // Handle clearing messages
  const handleClear = () => {
    setMessages([]);
    setEditingMessageId(null);
    setEditingText('');
  };

  // Handle starting edit mode
  const handleEdit = (messageId) => {
    const messageToEdit = messages.find(msg => msg.id === messageId);
    if (messageToEdit) {
      setEditingMessageId(messageId);
      setEditingText(messageToEdit.text);
    }
  };

  const handleSaveEdit = async (messageId) => {
    try {
      // Find the edited message and its index
      const messageIndex = messages.findIndex(msg => msg.id === messageId);
      const editedMessage = messages[messageIndex];
      
      if (messageIndex === -1 || editedMessage.sender !== 'user') return;
      
      // Remove all messages that came after this message
      const updatedMessages = messages.slice(0, messageIndex + 1);
      
      // Update the edited message
      updatedMessages[messageIndex] = {
        ...editedMessage,
        text: editingText
      };
      
      setMessages(updatedMessages);
      setEditingMessageId(null);
      setEditingText('');
      
      // Generate new response
      setLoading(true);
      
      const requestData = {
        model_name: selectedModel,
        quantization_bits: quantizationBits,
        quantization_type: quantizationType,
        input_text: editingText,
      };
  
      const postResponse = await makeApiRequest(API_BASE_URL.RUN_INFERENCE, {
        method: 'POST',
        body: JSON.stringify(requestData),
      });
  
      if (!postResponse || !postResponse.request_id) {
        throw new Error('Invalid response from server');
      }
  
      const results = await pollForResults(postResponse.request_id);
  
      const botMessage = {
        id: Date.now(),
        text: results.original.text,
        sender: 'bot',
        metrics: {
          original: {
            inferenceTime: results.original.inference_time,
            memoryUsage: results.original.memory_usage,
            perplexity: results.original.perplexity
          },
          quantized: {
            text: results.quantized.text,
            inferenceTime: results.quantized.inference_time,
            memoryUsage: results.quantized.memory_usage,
            perplexity: results.quantized.perplexity
          },
          comparison: results.comparison
        }
      };
  
      setMessages([...updatedMessages, botMessage]);
    } catch (error) {
      console.error('Error in handleSaveEdit:', error);
      const errorMessage = {
        id: Date.now(),
        text: `Error: ${error.message}. Please try again later.`,
        sender: 'bot',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // Handle settings changes
  const handleSettingsChange = async () => {
    const lastUserMessage = [...messages].reverse().find(msg => msg.sender === 'user');
    if (!lastUserMessage || loading) return;

    try {
      const results = await runInference(lastUserMessage.text);
      
      setMessages(prev => {
        const lastBotMessageIndex = prev.findIndex(msg => 
          msg.sender === 'bot' && 
          prev.indexOf(lastUserMessage) < prev.indexOf(msg)
        );

        if (lastBotMessageIndex === -1) return prev;

        return prev.map((msg, index) =>
          index === lastBotMessageIndex ? {
            ...msg,
            text: results.text,
            metrics: results.metrics
          } : msg
        );
      });
    } catch (error) {
      console.error('Error updating message after settings change:', error);
    }
  };

  // Settings change handlers
  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    handleSettingsChange();
  };

  const handleBitsChange = (e) => {
    setQuantizationBits(parseInt(e.target.value));
    handleSettingsChange();
  };

  const handleTypeChange = (e) => {
    setQuantizationType(e.target.value);
    handleSettingsChange();
  };

  // Message comparison component
  const MessageComparison = ({ message }) => {
    const [showStats, setShowStats] = useState(false);

    if (message.sender === 'user') {
      return (
        <div className="flex justify-end mb-4">
          <div className="group relative max-w-2xl">
            {editingMessageId === message.id ? (
              <div className="flex gap-2">
                <input
                  type="text"
                  value={editingText}
                  onChange={(e) => setEditingText(e.target.value)}
                  className="flex-1 bg-[#1D1D1F] text-white rounded-lg p-4 border border-gray-600"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleSaveEdit(message.id);
                    }
                  }}
                  autoFocus
                />
                <button
                  onClick={() => handleSaveEdit(message.id)}
                  className="p-2 bg-[#1D1D1F] text-white rounded-lg border border-gray-600 hover:bg-gray-500 transition-colors"
                >
                  <Check size={20} />
                </button>
              </div>
            ) : (
              <div className="bg-[#1D1D1F] text-white rounded-lg p-4 relative group">
                {message.text}
                <button
                  onClick={() => handleEdit(message.id)}
                  className="absolute right-2 top-2 p-1 text-gray-400 opacity-0 group-hover:opacity-100 hover:text-white hover:bg-gray-700 rounded transition-all"
                >
                  <Edit2 size={16} />
                </button>
              </div>
            )}
          </div>
        </div>
      );
    }

    return (
      <div className="relative w-full mb-4 bg-[#1D1D1F] p-6 rounded-lg">
        <div className="grid grid-cols-2 gap-4">
          {/* Original Text */}
          <div className="space-y-2">
            <h3 className="font-medium text-gray-200">Original Model</h3>
            <div className="bg-[#1D1D1F] rounded-lg p-4 text-white whitespace-pre-wrap">
              <ReactMarkdown className="prose prose-invert max-w-none">
                {message.metrics?.original?.text || message.text}
              </ReactMarkdown>
            </div>
          </div>

          {/* Quantized Text */}
          <div className="space-y-2">
            <h3 className="font-medium text-gray-200">Quantized Model</h3>
            <div className="bg-[#1D1D1F] rounded-lg p-4 text-white whitespace-pre-wrap">
              <ReactMarkdown className="prose prose-invert max-w-none">
                {message.metrics?.quantized?.text || message.text}
              </ReactMarkdown>
            </div>
          </div>

          {/* Stats Button */}
          {message.metrics?.comparison && (
            <div className="absolute top-4 right-4">
              <div className="relative">
                <button
                  onMouseEnter={() => setShowStats(true)}
                  onMouseLeave={() => setShowStats(false)}
                  className="p-2 bg-[#1D1D1F] hover:bg-gray-700 rounded-lg text-gray-300 hover:text-white transition-colors"
                >
                  <BarChart2 size={20} />
                </button>

                {/* Stats Popup */}
                {showStats && (
                  <div className="absolute right-0 mt-2 w-64 p-4 bg-[#1D1D1F] rounded-lg shadow-lg z-10">
                    <div className="space-y-2 text-sm text-white">
                      <p>Speed Improvement: {message.metrics.comparison.speed_diff.toFixed(2)}%</p>
                      <p>Memory Savings: {message.metrics.comparison.memory_savings.toFixed(2)}%</p>
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
    <div className="h-screen flex bg-[#1D1D1F]">
      {/* Settings Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} bg-black border-r border-gray-700 text-gray-300 transition-all duration-300 overflow-hidden`}>
        <div className="p-6 space-y-8">
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-white">Settings</h2>
            
            {/* Model Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">Model</label>
              <select 
                value={selectedModel}
                onChange={handleModelChange}
                className="w-full bg-[#1D1D1F] border border-gray-600 rounded-lg p-2 text-white"
              >
                <option value="Qwen/Qwen2.5-0.5B">Qwen/Qwen2.5-0.5B</option>
                <option value="Qwen/Qwen2.5-1.5B">Qwen/Qwen2.5-1.5B</option>
                <option value="meta-llama/Llama-3.2-1B">meta-llama/Llama-3.2-1B</option>
                <option value="meta-llama/Llama-3.2-3B">meta-llama/Llama-3.2-3B</option>
              </select>
            </div>

            {/* Quantization Bits Slider */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">Quantization Bits</label>
              <input
                type="range"
                min="2"
                max="20"
                value={quantizationBits}
                onChange={handleBitsChange}
                className="w-full h-2 bg-[#1D1D1F] rounded-lg appearance-none cursor-pointer"
              />
              <div className="text-center text-gray-300">{quantizationBits} bits</div>
            </div>

            {/* Quantization Type Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">Quantization Type</label>
              <select
                value={quantizationType}
                onChange={handleTypeChange}
                className="w-full bg-[#1D1D1F] border border-gray-600 rounded-lg p-2 text-white"
              >
                <option value="WASQ-OPT">WASQ-OPT</option>
                <option value="WASQ-LTH">WASQ-LTH</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Toggle Sidebar Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="absolute left-0 top-4 bg-[#1D1D1F] text-gray-400 p-2 rounded-r-lg hover:bg-gray-800 transition-colors"
      >
        {sidebarOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
      </button>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="relative flex-1 p-6 overflow-y-auto">
          <button 
            onClick={handleClear}
            className="absolute top-4 right-4 p-2 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <Trash2 size={20} />
          </button>
          <div className="space-y-6 pt-12">
            {messages.map((message) => (
              <MessageComparison key={message.id} message={message} />
            ))}
            {loading && (
              <div className="flex justify-center">
                <div className="bg-[#1D1D1F] rounded-lg p-4 text-white">
                  Processing request... This may take up to 2 minutes.
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-700 bg-[#1D1D1F] p-6">
          <div className="max-w-4xl mx-auto flex gap-4">
            <input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Type your message..."
              className="flex-1 p-4 bg-[#1D1D1F] text-white rounded-lg border border-gray-600 focus:border-gray-400 focus:ring-0 transition-colors"
              disabled={loading}
            />
            <button 
              onClick={handleSend}
              className="p-4 bg-[#1D1D1F] text-white rounded-lg hover:bg-gray-700 transition-colors disabled:bg-gray-700"
              disabled={loading}
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;