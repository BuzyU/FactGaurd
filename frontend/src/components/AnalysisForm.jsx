import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Link, FileText, Send } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const AnalysisForm = ({ onAnalysisStart, onAnalysisComplete }) => {
  const [inputType, setInputType] = useState('text');
  const [textContent, setTextContent] = useState('');
  const [urlContent, setUrlContent] = useState('');
  const [loading, setLoading] = useState(false);
  const { idToken } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!textContent.trim() && !urlContent.trim()) {
      toast.error('Please enter some text or a URL to analyze');
      return;
    }

    setLoading(true);
    onAnalysisStart();

    try {
      const payload = {};
      if (inputType === 'text') {
        payload.text = textContent.trim();
      } else {
        payload.url = urlContent.trim();
      }

      const response = await axios.post(
        'http://localhost:8000/analyze',
        payload,
        {
          headers: {
            'Authorization': `Bearer ${idToken}`,
            'Content-Type': 'application/json'
          }
        }
      );

      onAnalysisComplete(response.data);
      toast.success('Analysis completed successfully!');
    } catch (error) {
      console.error('Analysis error:', error);
      const errorMessage = error.response?.data?.detail || 'Analysis failed. Please try again.';
      toast.error(errorMessage);
      onAnalysisComplete(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setTextContent('');
    setUrlContent('');
  };

  const exampleTexts = [
    "Scientists have discovered that drinking 8 glasses of water daily can cure cancer completely.",
    "New study shows that vaccines contain microchips for government tracking.",
    "Climate change is a natural phenomenon and has nothing to do with human activities."
  ];

  const handleExampleClick = (example) => {
    setInputType('text');
    setTextContent(example);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Input Type Selector */}
      <div className="flex space-x-4">
        <button
          type="button"
          onClick={() => setInputType('text')}
          className={`flex items-center px-4 py-2 rounded-lg border transition-colors ${
            inputType === 'text'
              ? 'bg-primary-50 border-primary-200 text-primary-700'
              : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
          }`}
        >
          <FileText className="h-4 w-4 mr-2" />
          Text
        </button>
        <button
          type="button"
          onClick={() => setInputType('url')}
          className={`flex items-center px-4 py-2 rounded-lg border transition-colors ${
            inputType === 'url'
              ? 'bg-primary-50 border-primary-200 text-primary-700'
              : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
          }`}
        >
          <Link className="h-4 w-4 mr-2" />
          URL
        </button>
      </div>

      {/* Text Input */}
      {inputType === 'text' && (
        <div>
          <label htmlFor="text-content" className="block text-sm font-medium text-gray-700 mb-2">
            Enter text to analyze
          </label>
          <textarea
            id="text-content"
            value={textContent}
            onChange={(e) => setTextContent(e.target.value)}
            rows={6}
            className="input-field resize-none"
            placeholder="Paste the text content you want to fact-check..."
          />
          
          {/* Example Texts */}
          <div className="mt-3">
            <p className="text-xs text-gray-500 mb-2">Try these examples:</p>
            <div className="space-y-1">
              {exampleTexts.map((example, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => handleExampleClick(example)}
                  className="block text-xs text-primary-600 hover:text-primary-800 text-left"
                >
                  "{example.substring(0, 80)}..."
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* URL Input */}
      {inputType === 'url' && (
        <div>
          <label htmlFor="url-content" className="block text-sm font-medium text-gray-700 mb-2">
            Enter URL to analyze
          </label>
          <input
            id="url-content"
            type="url"
            value={urlContent}
            onChange={(e) => setUrlContent(e.target.value)}
            className="input-field"
            placeholder="https://example.com/article"
          />
          <p className="mt-1 text-xs text-gray-500">
            We'll extract and analyze the text content from the webpage
          </p>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex space-x-3">
        <button
          type="submit"
          disabled={loading || (!textContent.trim() && !urlContent.trim())}
          className="flex items-center btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Analyzing...
            </>
          ) : (
            <>
              <Send className="h-4 w-4 mr-2" />
              Analyze Content
            </>
          )}
        </button>
        
        <button
          type="button"
          onClick={handleClear}
          className="btn-secondary"
          disabled={loading}
        >
          Clear
        </button>
      </div>
    </form>
  );
};

export default AnalysisForm;
