import React, { useState } from 'react';
import { 
  TrendingUp, 
  Target, 
  Eye, 
  Info, 
  HelpCircle,
  CheckCircle,
  AlertTriangle,
  XCircle,
  BarChart3,
  Zap
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Separator } from './ui/separator';

interface F1ScoreDisplayProps {
  f1Score?: number;
  precision?: number;
  recall?: number;
  performanceExplanation?: string;
  performanceMetrics?: {
    f1_score: number;
    precision: number;
    recall: number;
    true_positives: number;
    false_positives: number;
    false_negatives: number;
    explanation: string;
  };
  predictedCount: number;
  correctedCount?: number | null;
  compact?: boolean;
  showTooltip?: boolean;
}

export function F1ScoreDisplay({
  f1Score,
  precision,
  recall,
  performanceExplanation,
  performanceMetrics,
  predictedCount,
  correctedCount,
  compact = false,
  showTooltip = true
}: F1ScoreDisplayProps) {
  const [showExplanationDialog, setShowExplanationDialog] = useState(false);

  // Get badge styling based on F1 score
  const getBadgeInfo = (score?: number) => {
    if (score === undefined || score === null) {
      return {
        level: 'No Feedback',
        color: 'text-gray-600',
        bgColor: 'bg-gray-100',
        borderColor: 'border-gray-300',
        icon: <HelpCircle className="h-4 w-4" />
      };
    } else if (score >= 90) {
      return {
        level: 'Excellent',
        color: 'text-green-800',
        bgColor: 'bg-green-100',
        borderColor: 'border-green-300',
        icon: <CheckCircle className="h-4 w-4 text-green-600" />
      };
    } else if (score >= 75) {
      return {
        level: 'Very Good',
        color: 'text-blue-800',
        bgColor: 'bg-blue-100',
        borderColor: 'border-blue-300',
        icon: <Target className="h-4 w-4 text-blue-600" />
      };
    } else if (score >= 60) {
      return {
        level: 'Good',
        color: 'text-yellow-800',
        bgColor: 'bg-yellow-100',
        borderColor: 'border-yellow-300',
        icon: <AlertTriangle className="h-4 w-4 text-yellow-600" />
      };
    } else {
      return {
        level: 'Needs Review',
        color: 'text-red-800',
        bgColor: 'bg-red-100',
        borderColor: 'border-red-300',
        icon: <XCircle className="h-4 w-4 text-red-600" />
      };
    }
  };

  const badgeInfo = getBadgeInfo(f1Score);

  // Compact version for cards and lists
  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <Badge className={`${badgeInfo.bgColor} ${badgeInfo.color} ${badgeInfo.borderColor} border`}>
          <span className="flex items-center gap-1">
            {badgeInfo.icon}
            {f1Score !== undefined ? `F1: ${f1Score.toFixed(0)}%` : badgeInfo.level}
          </span>
        </Badge>
        
        {showTooltip && f1Score !== undefined && (
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                <Info className="h-3 w-3" />
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-blue-500" />
                  F1 Score Breakdown
                </DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="p-2 bg-blue-50 rounded">
                    <p className="text-xs text-gray-600">F1 Score</p>
                    <p className="font-bold text-blue-600">{f1Score.toFixed(1)}%</p>
                  </div>
                  <div className="p-2 bg-green-50 rounded">
                    <p className="text-xs text-gray-600">Precision</p>
                    <p className="font-bold text-green-600">{precision?.toFixed(1) || 0}%</p>
                  </div>
                  <div className="p-2 bg-purple-50 rounded">
                    <p className="text-xs text-gray-600">Recall</p>
                    <p className="font-bold text-purple-600">{recall?.toFixed(1) || 0}%</p>
                  </div>
                </div>
                
                {performanceExplanation && (
                  <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
                    {performanceExplanation}
                  </p>
                )}
              </div>
            </DialogContent>
          </Dialog>
        )}
      </div>
    );
  }

  // Full detailed version
  return (
    <Card className={`${badgeInfo.borderColor} border-2`}>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center justify-between">
          <span className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-blue-500" />
            Performance Metrics
          </span>
          
          <div className="flex items-center gap-2">
            <Badge className={`${badgeInfo.bgColor} ${badgeInfo.color} ${badgeInfo.borderColor} border`}>
              <span className="flex items-center gap-1">
                {badgeInfo.icon}
                {badgeInfo.level}
              </span>
            </Badge>
            
            <Dialog open={showExplanationDialog} onOpenChange={setShowExplanationDialog}>
              <DialogTrigger asChild>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  <Info className="h-4 w-4" />
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle className="flex items-center gap-2">
                    <Zap className="h-6 w-6 text-blue-500" />
                    Why F1 Score is Better for Object Counting
                  </DialogTitle>
                </DialogHeader>
                <div className="space-y-6">
                  {/* Why F1 Score */}
                  <div>
                    <h3 className="font-semibold mb-3">🎯 Why F1 Score vs Simple Accuracy?</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-3 bg-blue-50 rounded-lg">
                        <h4 className="font-medium text-blue-800 mb-2">✅ F1 Score Benefits:</h4>
                        <ul className="text-sm text-blue-700 space-y-1">
                          <li>• Handles imbalanced data better</li>
                          <li>• Considers precision & recall</li>
                          <li>• Standard ML evaluation metric</li>
                          <li>• More nuanced than simple accuracy</li>
                        </ul>
                      </div>
                      <div className="p-3 bg-yellow-50 rounded-lg">
                        <h4 className="font-medium text-yellow-800 mb-2">⚠️ Simple Accuracy Issues:</h4>
                        <ul className="text-sm text-yellow-700 space-y-1">
                          <li>• Misleading with sparse objects</li>
                          <li>• Doesn't distinguish error types</li>
                          <li>• Less informative for counting</li>
                          <li>• Can hide model weaknesses</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                  
                  <Separator />
                  
                  {/* Formula Explanation */}
                  <div>
                    <h3 className="font-semibold mb-3">📐 F1 Score Formula</h3>
                    <div className="bg-gray-50 p-4 rounded-lg font-mono text-center">
                      <div className="text-lg mb-2">F1 = 2 × (Precision × Recall) / (Precision + Recall)</div>
                      <div className="text-sm text-gray-600">
                        <div>Precision = True Positives / (True Positives + False Positives)</div>
                        <div>Recall = True Positives / (True Positives + False Negatives)</div>
                      </div>
                    </div>
                  </div>
                  
                  <Separator />
                  
                  {/* Current Result Analysis */}
                  {performanceMetrics && (
                    <div>
                      <h3 className="font-semibold mb-3">📊 Current Result Analysis</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-3">
                          <div className="p-3 bg-green-50 rounded-lg">
                            <h4 className="font-medium text-green-800 mb-2">Object Detection Breakdown:</h4>
                            <div className="text-sm text-green-700 space-y-1">
                              <div>Correctly detected: {performanceMetrics.true_positives}</div>
                              <div>Over-detected: {performanceMetrics.false_positives}</div>
                              <div>Missed: {performanceMetrics.false_negatives}</div>
                            </div>
                          </div>
                          
                          <div className="p-3 bg-purple-50 rounded-lg">
                            <h4 className="font-medium text-purple-800 mb-2">Performance Scores:</h4>
                            <div className="text-sm text-purple-700 space-y-1">
                              <div>F1 Score: {performanceMetrics.f1_score.toFixed(1)}%</div>
                              <div>Precision: {performanceMetrics.precision.toFixed(1)}%</div>
                              <div>Recall: {performanceMetrics.recall.toFixed(1)}%</div>
                            </div>
                          </div>
                        </div>
                        
                        <div className="p-3 bg-blue-50 rounded-lg">
                          <h4 className="font-medium text-blue-800 mb-2">Interpretation:</h4>
                          <p className="text-sm text-blue-700">
                            {performanceMetrics.explanation}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {f1Score !== undefined ? (
          <>
            {/* Main F1 Score Display */}
            <div className="text-center p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
              <div className="flex items-center justify-center gap-2 mb-2">
                <TrendingUp className="h-6 w-6 text-blue-600" />
                <h3 className="text-lg font-semibold text-blue-800">F1 Score</h3>
              </div>
              <div className="text-4xl font-bold text-blue-600 mb-1">
                {f1Score.toFixed(1)}%
              </div>
              <div className="text-sm text-blue-700">
                {badgeInfo.level} Performance
              </div>
            </div>
            
            {/* Precision and Recall Breakdown */}
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
                <div className="flex items-center justify-center gap-1 mb-2">
                  <Target className="h-4 w-4 text-green-600" />
                  <span className="text-sm font-medium text-green-800">Precision</span>
                </div>
                <div className="text-2xl font-bold text-green-600 mb-1">
                  {precision?.toFixed(1) || 0}%
                </div>
                <div className="text-xs text-green-700">
                  Detection Accuracy
                </div>
                {precision !== undefined && (
                  <Progress value={precision} className="mt-2 h-2" />
                )}
              </div>
              
              <div className="text-center p-3 bg-purple-50 rounded-lg border border-purple-200">
                <div className="flex items-center justify-center gap-1 mb-2">
                  <Eye className="h-4 w-4 text-purple-600" />
                  <span className="text-sm font-medium text-purple-800">Recall</span>
                </div>
                <div className="text-2xl font-bold text-purple-600 mb-1">
                  {recall?.toFixed(1) || 0}%
                </div>
                <div className="text-xs text-purple-700">
                  Object Discovery
                </div>
                {recall !== undefined && (
                  <Progress value={recall} className="mt-2 h-2" />
                )}
              </div>
            </div>
            
            {/* Performance Explanation */}
            {performanceExplanation && (
              <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                <h4 className="font-medium mb-2 flex items-center gap-2">
                  <Info className="h-4 w-4 text-gray-600" />
                  Performance Insights
                </h4>
                <p className="text-sm text-gray-700">
                  {performanceExplanation}
                </p>
              </div>
            )}
            
            {/* Comparison with Counts */}
            <div className="grid grid-cols-2 gap-4 pt-2 border-t border-gray-200">
              <div className="text-center">
                <div className="text-sm text-gray-600 mb-1">AI Predicted</div>
                <div className="text-xl font-bold text-blue-600">{predictedCount}</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-600 mb-1">User Corrected</div>
                <div className="text-xl font-bold text-green-600">
                  {correctedCount !== null ? correctedCount : '—'}
                </div>
              </div>
            </div>
          </>
        ) : (
          /* No Feedback State */
          <div className="text-center p-6 bg-gray-50 rounded-lg border border-gray-200">
            <HelpCircle className="h-12 w-12 text-gray-400 mx-auto mb-3" />
            <h3 className="text-lg font-medium text-gray-600 mb-2">No Performance Data</h3>
            <p className="text-sm text-gray-500 mb-4">
              F1 Score will be calculated once user feedback is provided.
            </p>
            <div className="text-sm text-gray-600">
              <div>AI Predicted: <span className="font-semibold">{predictedCount} objects</span></div>
              <div>User Feedback: <span className="text-gray-400">Not provided</span></div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}



