import React from 'react';
import { AlertTriangle, CheckCircle, Info, ExternalLink, Lightbulb, Clock } from 'lucide-react';

const AnalysisResults = ({ result }) => {
  if (!result) return null;

  const getRiskIcon = (risk) => {
    switch (risk) {
      case 'high':
        return <AlertTriangle className="h-5 w-5 text-danger-600" />;
      case 'medium':
        return <Info className="h-5 w-5 text-warning-600" />;
      case 'low':
        return <CheckCircle className="h-5 w-5 text-success-600" />;
      default:
        return <Info className="h-5 w-5 text-gray-600" />;
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

  const getRiskMessage = (risk) => {
    switch (risk) {
      case 'high':
        return 'High risk of misinformation detected. Exercise extreme caution and verify through multiple reliable sources.';
      case 'medium':
        return 'Moderate concerns identified. Additional verification recommended before sharing or believing this content.';
      case 'low':
        return 'Content appears reliable, but always maintain healthy skepticism and cross-reference important claims.';
      default:
        return 'Unable to determine risk level. Please verify manually.';
    }
  };

  return (
    <div className="space-y-6">
      {/* Overall Risk Assessment */}
      <div className={`card p-6 ${getRiskColor(result.overall_risk)}`}>
        <div className="flex items-start">
          {getRiskIcon(result.overall_risk)}
          <div className="ml-3">
            <h3 className="text-lg font-semibold capitalize mb-2">
              {result.overall_risk} Risk Level
            </h3>
            <p className="text-sm mb-3">
              {getRiskMessage(result.overall_risk)}
            </p>
            {result.processing_time && (
              <div className="flex items-center text-xs opacity-75">
                <Clock className="h-3 w-3 mr-1" />
                Analyzed in {result.processing_time.toFixed(2)} seconds
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Key Findings */}
      {result.explanations && result.explanations.length > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Findings</h3>
          <ul className="space-y-2">
            {result.explanations.map((explanation, index) => (
              <li key={index} className="flex items-start">
                <div className="w-2 h-2 bg-primary-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                <span className="text-sm text-gray-700">{explanation}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Claims Analysis */}
      {result.claims && result.claims.length > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Extracted Claims ({result.claims.length})
          </h3>
          <div className="space-y-3">
            {result.claims.map((claim, index) => (
              <div key={claim.id || index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="text-sm text-gray-800 mb-2">{claim.text}</p>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>Type: {claim.type || 'factual'}</span>
                      {claim.confidence && (
                        <span>Confidence: {(claim.confidence * 100).toFixed(0)}%</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Supporting Evidence */}
      {result.evidence && result.evidence.length > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Supporting Evidence ({result.evidence.length})
          </h3>
          <div className="space-y-4">
            {result.evidence.map((evidence, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900 mb-2">{evidence.title}</h4>
                    <p className="text-sm text-gray-600 mb-3">{evidence.snippet}</p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>{evidence.domain}</span>
                        {evidence.date && <span>{evidence.date}</span>}
                        {evidence.credibility_score && (
                          <span>
                            Credibility: {(evidence.credibility_score * 100).toFixed(0)}%
                          </span>
                        )}
                        {evidence.relevance_score && (
                          <span>
                            Relevance: {(evidence.relevance_score * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                      <a
                        href={evidence.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center text-primary-600 hover:text-primary-800 text-sm"
                      >
                        <ExternalLink className="h-3 w-3 mr-1" />
                        View Source
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Educational Tips */}
      {result.teach_tips && result.teach_tips.length > 0 && (
        <div className="card p-6 bg-blue-50 border-blue-200">
          <div className="flex items-start">
            <Lightbulb className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
            <div>
              <h3 className="text-lg font-semibold text-blue-900 mb-4">
                How to Verify This Information
              </h3>
              <ul className="space-y-2">
                {result.teach_tips.map((tip, index) => (
                  <li key={index} className="flex items-start">
                    <CheckCircle className="h-4 w-4 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
                    <span className="text-sm text-blue-800">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysisResults;
