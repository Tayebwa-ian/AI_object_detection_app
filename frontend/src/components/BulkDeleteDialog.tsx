import React, { useEffect, useState } from 'react';
import { 
  X, 
  Trash2, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Loader2,
  FileX,
  Database,
  HardDrive
} from 'lucide-react';
import api from '../services/api';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Separator } from './ui/separator';
import { Progress } from './ui/progress';

interface BulkDeleteDialogProps {
  isOpen: boolean;
  onClose: () => void;
  selectedResults: Array<{
    id: string;
    object_type: string;
    predicted_count: number;
    corrected_count: number | null;
    image_path: string;
    created_at: string;
  }>;
  onDeleteComplete: (deletedIds: string[]) => void;
}

interface DeletionProgress {
  total: number;
  completed: number;
  failed: number;
  current?: string;
  deletedIds: string[];
  failures: Array<{ id: string; reason: string }>;
}

export function BulkDeleteDialog({ 
  isOpen, 
  onClose, 
  selectedResults, 
  onDeleteComplete 
}: BulkDeleteDialogProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [progress, setProgress] = useState<DeletionProgress | null>(null);
  const [error, setError] = useState<string>('');
  const [showDetails, setShowDetails] = useState(false);

  // Keep IDs readable in a compact list
  const shortId = (id: string) => (id && id.length > 12 ? `${id.slice(0, 6)}…${id.slice(-4)}` : id);

  // Reset internal state whenever the dialog is opened to avoid stale UI
  useEffect(() => {
    if (isOpen) {
      setIsDeleting(false);
      setProgress(null);
      setError('');
      setShowDetails(false);
    }
  }, [isOpen]);
  
  const handleBulkDelete = async () => {
    if (selectedResults.length === 0) return;
    
    setIsDeleting(true);
    setError('');
    setProgress({
      total: selectedResults.length,
      completed: 0,
      failed: 0,
      deletedIds: [],
      failures: []
    });
    
    try {
      const resultIds = selectedResults.map(result => result.id);
      
      setProgress(prev => prev ? { ...prev, current: 'Initiating bulk deletion...' } : null);
      
      const response = await api.bulkDeleteResults(resultIds);
      
      if (response.success) {
        setProgress(prev => prev ? {
          ...prev,
          completed: response.deleted_count,
          failed: response.failed_count,
          deletedIds: response.deleted_result_ids,
          failures: response.failures || [],
          current: 'Deletion completed'
        } : null);
        
        // Close quickly for snappier UX
        setTimeout(() => {
          onDeleteComplete(response.deleted_result_ids);
          onClose();
        }, 200);
        
      } else {
        throw new Error('Bulk deletion failed');
      }
      
    } catch (err) {
      console.error('Bulk deletion error:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete selected results');
      setProgress(prev => prev ? { ...prev, current: 'Deletion failed' } : null);
    } finally {
      setIsDeleting(false);
    }
  };
  
  const calculateDataToDelete = () => {
    const totalImages = selectedResults.length;
    const totalObjects = selectedResults.reduce((sum, result) => sum + result.predicted_count, 0);
    const withFeedback = selectedResults.filter(result => result.corrected_count !== null).length;
    
    return { totalImages, totalObjects, withFeedback };
  };
  
  const getProgressPercentage = () => {
    if (!progress || progress.total === 0) return 0;
    return Math.round((progress.completed + progress.failed) / progress.total * 100);
  };
  
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };
  
  if (!isOpen) return null;
  
  const { totalImages, totalObjects, withFeedback } = calculateDataToDelete();
  
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <div className="flex items-start justify-between gap-3">
            <DialogTitle className="flex items-center gap-2">
              <Trash2 className="h-5 w-5 text-red-600" />
              <span>Bulk Delete Confirmation</span>
            </DialogTitle>
            <Badge variant="secondary" className="shrink-0">
              {selectedResults.length} {selectedResults.length === 1 ? 'item' : 'items'}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground">
            Review the impact below before deleting. This action cannot be undone.
          </p>
        </DialogHeader>

        {error && (
          <Alert className="border-red-200 bg-red-50">
            <AlertTriangle className="h-4 w-4 text-red-500" />
            <AlertDescription className="text-red-700">{error}</AlertDescription>
          </Alert>
        )}

        {/* Deletion Progress */}
        {progress && (
          <Card className="border-blue-200 bg-blue-50">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                {isDeleting ? (
                  <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
                ) : progress.failed > 0 ? (
                  <XCircle className="h-5 w-5 text-red-500" />
                ) : (
                  <CheckCircle className="h-5 w-5 text-green-500" />
                )}
                Deletion Progress
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>{getProgressPercentage()}%</span>
                </div>
                <Progress value={getProgressPercentage()} className="h-2" />
              </div>
              
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="p-3 bg-white rounded-lg border">
                  <p className="text-sm text-gray-600">Completed</p>
                  <p className="text-xl font-bold text-green-600">{progress.completed}</p>
                </div>
                <div className="p-3 bg-white rounded-lg border">
                  <p className="text-sm text-gray-600">Failed</p>
                  <p className="text-xl font-bold text-red-600">{progress.failed}</p>
                </div>
                <div className="p-3 bg-white rounded-lg border">
                  <p className="text-sm text-gray-600">Total</p>
                  <p className="text-xl font-bold text-blue-600">{progress.total}</p>
                </div>
              </div>
              
              {progress.current && (
                <div className="text-sm text-gray-600 text-center italic">
                  {progress.current}
                </div>
              )}
              
              {progress.failures.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowDetails(!showDetails)}
                  className="w-full"
                >
                  {showDetails ? 'Hide' : 'Show'} Failure Details
                </Button>
              )}
              
              {showDetails && progress.failures.length > 0 && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <h4 className="font-medium text-red-800 mb-2">Failed Deletions:</h4>
                  <div className="space-y-1">
                    {progress.failures.map((failure, index) => (
                      <div key={index} className="text-sm text-red-700">
                        ID {failure.id}: {failure.reason}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
        
        {/* Deletion Warning */}
        {!progress && (
          <div className="space-y-6">
            <Alert className="border-red-200 bg-red-50">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <AlertDescription className="text-red-700">
                Permanently remove the selected records and their image files.
              </AlertDescription>
            </Alert>
            
            {/* Summary of what will be deleted */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">What will be deleted:</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
                    <FileX className="h-8 w-8 text-red-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 mb-1">Image Files</p>
                    <p className="text-2xl font-bold text-red-600">{totalImages}</p>
                  </div>
                  
                  <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
                    <Database className="h-8 w-8 text-red-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 mb-1">Database Records</p>
                    <p className="text-2xl font-bold text-red-600">{totalImages}</p>
                  </div>
                  
                  <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
                    <HardDrive className="h-8 w-8 text-red-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 mb-1">User Feedback</p>
                    <p className="text-2xl font-bold text-red-600">{withFeedback}</p>
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-2">
                  <h4 className="font-medium">Selected Items:</h4>
                  <div className="max-h-48 overflow-y-auto space-y-2 pr-1">
                    {selectedResults.map((result) => (
                      <div key={result.id} className="flex items-center justify-between p-2 bg-gray-50 rounded border">
                        <div className="flex items-center gap-3">
                          <Badge title={result.id} variant="secondary" className="font-mono text-xs max-w-[220px] truncate">
                            ID {shortId(result.id)}
                          </Badge>
                          <span className="font-medium capitalize">{result.object_type}</span>
                          <span className="text-sm text-gray-600">
                            {result.predicted_count} {result.predicted_count === 1 ? 'object' : 'objects'}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500">
                          {formatDate(result.created_at)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <div className="flex items-center justify-end gap-3">
              <Button
                variant="outline"
                onClick={onClose}
                disabled={isDeleting}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleBulkDelete}
                disabled={isDeleting || selectedResults.length === 0}
                className="flex items-center gap-2"
              >
                {isDeleting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4" />
                )}
                {isDeleting ? 'Deleting...' : `Delete ${selectedResults.length} ${selectedResults.length === 1 ? 'Item' : 'Items'}`}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
