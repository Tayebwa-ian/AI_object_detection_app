import React, { useState, useEffect } from 'react';
import { 
  X, 
  Trash2, 
  Edit, 
  Check, 
  AlertTriangle, 
  TrendingUp, 
  Clock, 
  Eye, 
  Calendar,
  Target,
  BarChart3,
  Camera,
  Save,
  RefreshCw
} from 'lucide-react';
import api, { ApiObjectType } from '../services/api';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Separator } from './ui/separator';
import { Alert, AlertDescription } from './ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { F1ScoreDisplay } from './F1ScoreDisplay';

interface ResultDetailsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  resultId: string | null;
  onDelete?: (resultId: string) => void;
  onUpdate?: (resultId: string) => void;
}

interface DetailedResult {
  id: string;
  predicted_count: number;
  corrected_count: number | null;
  object_type: string;
  object_type_id: number;
  image_path: string;
  description: string;
  created_at: string;
  updated_at: string;
  processing_time?: number;
  total_segments?: number;
  accuracy?: number;
  f1_score?: number;
  precision?: number;
  recall?: number;
  performance_explanation?: string;
  performance_metrics?: {
    f1_score: number;
    precision: number;
    recall: number;
    true_positives: number;
    false_positives: number;
    false_negatives: number;
    explanation: string;
  };
  difference?: number;
  has_feedback: boolean;
}

export function ResultDetailsDialog({ 
  isOpen, 
  onClose, 
  resultId, 
  onDelete, 
  onUpdate 
}: ResultDetailsDialogProps) {
  const [result, setResult] = useState<DetailedResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [deleting, setDeleting] = useState(false);
  const [updating, setUpdating] = useState(false);
  
  // Editing state
  const [isEditing, setIsEditing] = useState(false);
  const [editFeedback, setEditFeedback] = useState<number>(0);
  const [editObjectType, setEditObjectType] = useState<string>('');
  const [objectTypes, setObjectTypes] = useState<ApiObjectType[]>([]);
  
  // Load object types for editing
  useEffect(() => {
    const loadObjectTypes = async () => {
      try {
        const types = await api.getObjectTypes();
        setObjectTypes(types);
      } catch (error) {
        console.error('Failed to load object types:', error);
      }
    };
    
    if (isOpen) {
      loadObjectTypes();
    }
  }, [isOpen]);
  
  // Load result details when dialog opens
  useEffect(() => {
    if (isOpen && resultId) {
      loadResultDetails();
    }
  }, [isOpen, resultId]);
  
  const loadResultDetails = async () => {
    if (!resultId) return;
    
    setLoading(true);
    setError('');
    
    try {
      const response = await api.getResultDetails(resultId);
      if (response.success) {
        const resultData = response.result;
        setResult(resultData);
        
        // Initialize editing values
        setEditFeedback(resultData.corrected_count || resultData.predicted_count);
        setEditObjectType(resultData.object_type);
      } else {
        setError('Failed to load result details');
      }
    } catch (err) {
      console.error('Error loading result details:', err);
      setError(err instanceof Error ? err.message : 'Failed to load result details');
    } finally {
      setLoading(false);
    }
  };
  
  const handleDelete = async () => {
    if (!result || deleting) return;
    
    if (!window.confirm(`Are you sure you want to delete this result? This action cannot be undone.`)) {
      return;
    }
    
    setDeleting(true);
    setError('');
    
    try {
      console.log(`🗑️ Deleting result ${result.id} from detail dialog...`);
      
      const response = await api.deleteResult(result.id);
      console.log(`✅ Delete API response:`, response);
      
      if (response && response.success) {
        console.log(`🎉 Successfully deleted result ${result.id}`);
        onDelete?.(result.id);
        onClose();
      } else {
        throw new Error(response?.message || 'Delete operation failed');
      }
    } catch (err) {
      console.error('❌ Error deleting result:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete result';
      setError(errorMessage);
      
      // Don't close dialog on error so user can retry or see the error
    } finally {
      setDeleting(false);
    }
  };
  
  const handleUpdate = async () => {
    if (!result || updating) return;
    
    setUpdating(true);
    
    try {
      await api.updateResultFeedback(result.id, editFeedback, editObjectType);
      
      // Reload the result to get updated data
      await loadResultDetails();
      
      setIsEditing(false);
      onUpdate?.(result.id);
    } catch (err) {
      console.error('Error updating feedback:', err);
      setError(err instanceof Error ? err.message : 'Failed to update feedback');
    } finally {
      setUpdating(false);
    }
  };
  
  const startEditing = () => {
    if (result) {
      setEditFeedback(result.corrected_count || result.predicted_count);
      setEditObjectType(result.object_type);
      setIsEditing(true);
    }
  };
  
  const cancelEditing = () => {
    setIsEditing(false);
    if (result) {
      setEditFeedback(result.corrected_count || result.predicted_count);
      setEditObjectType(result.object_type);
    }
  };
  
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };
  
  const getF1ScoreBadge = (f1Score?: number) => {
    if (f1Score === undefined) {
      return <Badge variant="outline">No Feedback</Badge>;
    }
    
    if (f1Score >= 90) {
      return <Badge className="bg-green-100 text-green-800 border-green-300">Excellent</Badge>;
    } else if (f1Score >= 75) {
      return <Badge className="bg-blue-100 text-blue-800 border-blue-300">Very Good</Badge>;
    } else if (f1Score >= 60) {
      return <Badge className="bg-yellow-100 text-yellow-800 border-yellow-300">Good</Badge>;
    } else {
      return <Badge className="bg-red-100 text-red-800 border-red-300">Needs Review</Badge>;
    }
  };
  
  if (!isOpen) return null;
  
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="w-screen h-screen max-w-none max-h-none rounded-none p-0 overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <Camera className="h-6 w-6 text-blue-500" />
            <span>Result Details</span>
            {result && getF1ScoreBadge(result.f1_score)}
          </DialogTitle>
        </DialogHeader>
        
        {loading ? (
          <div className="flex items-center justify-center p-8">
            <div className="processing-spinner"></div>
            <span className="ml-3 text-gray-600">Loading result details...</span>
          </div>
        ) : error ? (
          <Alert className="border-red-200 bg-red-50">
            <AlertTriangle className="h-4 w-4 text-red-500" />
            <AlertDescription className="text-red-700">{error}</AlertDescription>
          </Alert>
        ) : result ? (
          <div className="space-y-6">
            {/* Header Actions */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-semibold capitalize">
                  {result.object_type} Detection
                </h2>
                <div className="flex items-center gap-1 text-sm text-gray-500">
                  <Calendar className="h-4 w-4" />
                  {formatDate(result.created_at)}
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                {!isEditing ? (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={startEditing}
                      className="flex items-center gap-2"
                    >
                      <Edit className="h-4 w-4" />
                      Edit Feedback
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleDelete}
                      disabled={deleting}
                      className="flex items-center gap-2 text-red-600 hover:text-red-700 hover:bg-red-50"
                    >
                      <Trash2 className="h-4 w-4" />
                      {deleting ? 'Deleting...' : 'Delete'}
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={cancelEditing}
                      className="flex items-center gap-2"
                    >
                      <X className="h-4 w-4" />
                      Cancel
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleUpdate}
                      disabled={updating}
                      className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
                    >
                      {updating ? (
                        <RefreshCw className="h-4 w-4 animate-spin" />
                      ) : (
                        <Save className="h-4 w-4" />
                      )}
                      {updating ? 'Updating...' : 'Save Changes'}
                    </Button>
                  </>
                )}
              </div>
            </div>
            
            <Separator />
            
            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Image Preview */}
              <div className="lg:col-span-1">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Camera className="h-5 w-5" />
                      Image Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="aspect-square rounded-lg bg-gray-100 flex items-center justify-center overflow-hidden">
                      {result.image_path ? (
                        <img 
                          src={result.image_path && result.image_path.startsWith('media/') ? `/${result.image_path}` : result.image_path}
                          alt="Analyzed image"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <Camera className="h-12 w-12 text-gray-400" />
                      )}
                    </div>
                    
                    {/* Technical Details */}
                    <div className="space-y-2 text-sm">
                      {result.processing_time && (
                        <div className="flex items-center justify-between">
                          <span className="text-gray-600 flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            Processing Time
                          </span>
                          <span className="font-medium">{result.processing_time.toFixed(1)}s</span>
                        </div>
                      )}
                      {result.total_segments && (
                        <div className="flex items-center justify-between">
                          <span className="text-gray-600 flex items-center gap-1">
                            <Eye className="h-4 w-4" />
                            Segments Analyzed
                          </span>
                          <span className="font-medium">{result.total_segments}</span>
                        </div>
                      )}
                      {result.description && (
                        <div>
                          <span className="text-gray-600 block mb-1">Description:</span>
                          <span className="text-sm">{result.description}</span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              {/* Analysis Results */}
              <div className="lg:col-span-2 space-y-6">
                {/* F1 Score Performance Metrics */}
                <F1ScoreDisplay
                  f1Score={result.f1_score}
                  precision={result.precision}
                  recall={result.recall}
                  performanceExplanation={result.performance_explanation}
                  performanceMetrics={result.performance_metrics}
                  predictedCount={result.predicted_count}
                  correctedCount={result.corrected_count}
                  compact={false}
                  showTooltip={true}
                />
                
                {/* Quick Edit Panel (when editing) */}
                {isEditing && (
                  <Card className="border-blue-200 bg-blue-50">
                    <CardHeader>
                      <CardTitle className="text-lg text-blue-800">Edit Feedback</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm font-medium text-blue-800 mb-2 block">Corrected Count:</label>
                          <Input
                            type="number"
                            value={editFeedback}
                            onChange={(e) => setEditFeedback(parseInt(e.target.value) || 0)}
                            className="text-center text-xl font-bold h-12"
                            min="0"
                          />
                        </div>
                        <div>
                          <label className="text-sm font-medium text-blue-800 mb-2 block">Object Type:</label>
                          <Select value={editObjectType} onValueChange={setEditObjectType}>
                            <SelectTrigger className="h-12">
                              <SelectValue placeholder="Select object type" />
                            </SelectTrigger>
                            <SelectContent>
                              {objectTypes.map((type) => (
                                <SelectItem key={type.id} value={type.name}>
                                  <span className="capitalize">{type.name}</span>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
                

                
                {/* Analysis Summary */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Analysis Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Object Type:</span>
                        <span className="ml-2 font-medium capitalize">{result.object_type}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Has Feedback:</span>
                        <span className="ml-2">
                          {result.has_feedback ? (
                            <Badge className="bg-green-100 text-green-800">Yes</Badge>
                          ) : (
                            <Badge variant="outline">No</Badge>
                          )}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Created:</span>
                        <span className="ml-2 font-medium">{formatDate(result.created_at)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Last Updated:</span>
                        <span className="ml-2 font-medium">{formatDate(result.updated_at)}</span>
                      </div>
                    </div>
                    
                    {result.f1_score !== undefined && (
                      <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
                        <h4 className="font-medium mb-2 flex items-center gap-2">
                          <TrendingUp className="h-4 w-4 text-blue-600" />
                          F1 Score Performance Insights
                        </h4>
                        <p className="text-sm text-gray-700">
                          {result.performance_explanation || 
                            (result.f1_score >= 90 
                              ? "🎯 Excellent F1 Score! The AI model shows outstanding precision and recall balance."
                              : result.f1_score >= 75
                              ? "✅ Very good F1 Score with strong precision and recall performance."
                              : result.f1_score >= 60
                              ? "👍 Good F1 Score, minor improvements in object detection accuracy possible."
                              : "🔄 F1 Score indicates room for improvement in precision or recall balance."
                            )
                          }
                        </p>
                        <div className="mt-3 text-xs text-blue-600 bg-white/50 p-2 rounded">
                          <strong>Why F1 Score?</strong> Better than simple accuracy for object counting as it considers both precision (avoiding false detections) and recall (finding all objects).
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
