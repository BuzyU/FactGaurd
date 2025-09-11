import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import AnalysisForm from './AnalysisForm';
import AnalysisResults from './AnalysisResults';
import { Shield, AlertTriangle, CheckCircle, Info } from 'lucide-react';

const Dashboard = () => {
  const { user } = useAuth();
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalysisComplete = (result) => {
    setAnalysisResult(result);
    setLoading(false);
  };

  const handleAnalysisStart = () => {
    setLoading(true);
    setAnalysisResult(null);
  };

  const getRiskIcon = (risk) => {
    switch (risk) {
      case 'high':
        return <AlertTriangle className="h-6 w-6 text-danger-600" />;
      case 'medium':
        return <Info className="h-6 w-6 text-warning-600" />;
      case 'low':
        return <CheckCircle className="h-6 w-6 text-success-600" />;
      default:
        return <Shield className="h-6 w-6 text-gray-600" />;
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'high':
        return 'risk-high';
      case 'medium':
        return 'risk-medium';
      case 'low':
        return 'risk-low';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">FactGuard Dashboard</h1>
        <p className="mt-2 text-gray-600">
          Analyze content for misinformation using AI-powered detection
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Analysis Form */}
        <div className="lg:col-span-2">
          <div className="card p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Analyze Content
            </h2>
            <AnalysisForm 
              onAnalysisStart={handleAnalysisStart}
              onAnalysisComplete={handleAnalysisComplete}
            />
          </div>

          {/* Loading State */}
          {loading && (
            <div className="card p-6 mt-6">
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mr-3"></div>
                <span className="text-gray-600">Analyzing content...</span>
              </div>
            </div>
          )}

          {/* Analysis Results */}
          {analysisResult && !loading && (
            <div className="mt-6">
              <AnalysisResults result={analysisResult} />
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Risk Summary */}
          {analysisResult && (
            <div className={`card p-6 ${getRiskColor(analysisResult.overall_risk)}`}>
              <div className="flex items-center mb-3">
                {getRiskIcon(analysisResult.overall_risk)}
                <h3 className="ml-2 text-lg font-semibold capitalize">
                  {analysisResult.overall_risk} Risk
                </h3>
              </div>
              <p className="text-sm">
                Analysis completed in {analysisResult.processing_time?.toFixed(2)}s
              </p>
            </div>
          )}

          {/* Educational Tips */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              How to Verify Information
            </h3>
            <ul className="space-y-3 text-sm text-gray-600">
              <li className="flex items-start">
                <CheckCircle className="h-4 w-4 text-success-500 mt-0.5 mr-2 flex-shrink-0" />
                Check multiple reliable sources
              </li>
              <li className="flex items-start">
                <CheckCircle className="h-4 w-4 text-success-500 mt-0.5 mr-2 flex-shrink-0" />
                Look for author credentials
              </li>
              <li className="flex items-start">
                <CheckCircle className="h-4 w-4 text-success-500 mt-0.5 mr-2 flex-shrink-0" />
                Verify publication dates
              </li>
              <li className="flex items-start">
                <CheckCircle className="h-4 w-4 text-success-500 mt-0.5 mr-2 flex-shrink-0" />
                Cross-reference with fact-checkers
              </li>
              <li className="flex items-start">
                <CheckCircle className="h-4 w-4 text-success-500 mt-0.5 mr-2 flex-shrink-0" />
                Be skeptical of emotional language
              </li>
            </ul>
          </div>

          {/* Quick Stats */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Quick Stats
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Analyses Today</span>
                <span className="text-sm font-medium">-</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">High Risk Detected</span>
                <span className="text-sm font-medium">-</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Sources Checked</span>
                <span className="text-sm font-medium">-</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
