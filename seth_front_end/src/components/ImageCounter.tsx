import React, { useState, useCallback, useEffect } from 'react';
import { Upload, X, Image as ImageIcon, AlertCircle, History, RefreshCw, Trash2, Zap, Cpu, Activity, BarChart3 } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from './ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { ProcessedImage } from './ResultsDashboard';
import { ImageHistory } from './ImageHistory';
import { SimpleProcessingDialog } from './SimpleProcessingDialog';
import { BatchProcessingDialog } from './BatchProcessingDialog';
import { ResultsDashboard } from './ResultsDashboard';
import { CompletionDialog } from './CompletionDialog';
import api from '../services/api';

interface UploadedImage {
  id: string;
  file: File;
  url: string;
  elementCount?: number;
  isProcessing?: boolean;
  resultId?: string;
  processingError?: string;
}

export function ImageCounter() {
  const [images, setImages] = useState<UploadedImage[]>([]);
  const [processedImages, setProcessedImages] = useState<ProcessedImage[]>([]);
  const [selectedObjectType, setSelectedObjectType] = useState<string>('');
  const [objectTypes, setObjectTypes] = useState<string[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showProcessingDialog, setShowProcessingDialog] = useState(false);
  const [showBatchProcessingDialog, setShowBatchProcessingDialog] = useState(false);
  const [showCompletionDialog, setShowCompletionDialog] = useState(false);
  const [apiError, setApiError] = useState<string>('');
  const [apiStatus, setApiStatus] = useState<'checking' | 'healthy' | 'error'>('checking');
  
  // Enhanced error handling
  const [errorDialog, setErrorDialog] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    details?: string;
    imageId?: string;
    canRetry?: boolean;
  }>({
    isOpen: false,
    title: '',
    message: '',
    canRetry: false
  });

  // Initialize API connection and fetch object types
  useEffect(() => {
    const initializeAPI = async () => {
      try {
        setApiStatus('checking');
        setApiError('');
        
        // Add a small delay to ensure backend is ready
        await new Promise(resolve => setTimeout(resolve, 500));

        // Check API health
        const healthCheck = await api.healthCheck();
        
        if (healthCheck.status === 'healthy' && healthCheck.pipeline_available) {
          setApiStatus('healthy');
          
          // Fetch available object types
          try {
            const objectTypeNames = await api.getObjectTypeNames();
            setObjectTypes(objectTypeNames);
            // Set default selection to first object type
            if (objectTypeNames.length > 0) {
              setSelectedObjectType(objectTypeNames[0]);
            }
          } catch (error) {
            console.error('Failed to fetch object types:', error);
            // Use fallback object types
            const fallbackTypes = ['car', 'person', 'dog', 'cat', 'tree', 'building'];
            setObjectTypes(fallbackTypes);
            setSelectedObjectType(fallbackTypes[0]);
          }
        } else {
          setApiStatus('error');
          setApiError(healthCheck.message || 'AI pipeline not available');
        }
      } catch (error) {
        console.error('Failed to initialize API:', error);
        setApiStatus('error');
        setApiError(error instanceof Error ? error.message : 'Failed to connect to backend');
      }
    };

    initializeAPI();
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files).filter((file: File) => 
      file.type.startsWith('image/')
    );
    
    handleFiles(files);
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files) as File[];
      handleFiles(files);
    }
  }, []);

  const handleFiles = (files: File[]) => {
    const newImages: UploadedImage[] = files.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      url: URL.createObjectURL(file),
    }));
    
    setImages(prev => [...prev, ...newImages]);
  };

  const removeImage = useCallback((id: string) => {
    setImages(prev => {
      const image = prev.find(img => img.id === id);
      if (image) {
        URL.revokeObjectURL(image.url);
      }
      return prev.filter(img => img.id !== id);
    });
  }, []);

  // Enhanced error handling functions
  const showErrorDialog = (title: string, message: string, details?: string, imageId?: string, canRetry = false) => {
    setErrorDialog({
      isOpen: true,
      title,
      message,
      details,
      imageId,
      canRetry
    });
  };

  const closeErrorDialog = () => {
    setErrorDialog(prev => ({ ...prev, isOpen: false }));
  };

  const retryImageProcessing = async (imageId: string) => {
    const image = images.find(img => img.id === imageId);
    if (!image) return;

    closeErrorDialog();
    
    // Mark image as processing
    setImages(prev => prev.map(img => 
      img.id === imageId 
        ? { ...img, isProcessing: true, processingError: undefined }
        : img
    ));

    try {
      console.log(`🔄 Retrying analysis for: ${image.file.name}`);
      
      const result = await api.countObjects(
        image.file,
        selectedObjectType,
        `Count ${selectedObjectType} objects in this image`
      );
      
      const totalCount = result.predicted_count || 0;
      
      // Update image with results
      setImages(prev => prev.map(img => 
        img.id === imageId 
          ? { 
              ...img, 
              elementCount: totalCount,
              isProcessing: false,
              resultId: result.result_id,
              processingError: undefined
            }
          : img
      ));

      // Add to processed images
      setProcessedImages(prev => [...prev, {
        id: image.id,
        file: image.file,
        url: image.url,
        objects: [{
          type: result.object_type,
          count: result.predicted_count
        }],
        resultId: result.result_id,
        processingTime: result.processing_time,
        totalSegments: result.total_segments
      }]);

    } catch (error) {
      console.error(`❌ Retry failed for ${image.file.name}:`, error);
      
      const errorMessage = error instanceof Error ? error.message : 'Analysis failed';
      const isGPUError = errorMessage.toLowerCase().includes('memory') || 
                        errorMessage.toLowerCase().includes('cuda') || 
                        errorMessage.toLowerCase().includes('allocator');
      
      setImages(prev => prev.map(img => 
        img.id === imageId 
          ? { ...img, isProcessing: false, processingError: errorMessage }
          : img
      ));

      // Show detailed error dialog
      showErrorDialog(
        isGPUError ? '🚀 GPU Memory Error' : '❌ Processing Error',
        isGPUError 
          ? 'Not enough GPU memory to process this image. Try removing other images or the system will fall back to CPU.'
          : 'Failed to analyze this image. You can try again or remove it.',
        errorMessage,
        imageId,
        true
      );
    }
  };

  const processImages = () => {
    if (images.length === 0 || apiStatus !== 'healthy') {
      setApiError('Please upload images to analyze');
      return;
    }
    
    if (!selectedObjectType) {
      setApiError('Please select an object type to detect');
      return;
    }
    
    // Clear any previous errors
    setApiError('');
    
    // Use batch processing for multiple images, simple processing for single image
    if (images.length > 1) {
      setShowBatchProcessingDialog(true);
    } else {
      setShowProcessingDialog(true);
    }
  };

  const handleProcessingComplete = (results: ProcessedImage[]) => {
    console.log('🎉 Processing completed successfully:', results);
    
    // Update images with results
    const updatedImages = images.map(img => {
      const result = results.find(r => r.file.name === img.file.name);
      if (result) {
        const totalCount = result.objects.reduce((sum, obj) => sum + obj.count, 0);
        return {
          ...img,
          elementCount: totalCount,
          resultId: result.resultId,
          processingError: undefined
        };
      }
      return img;
    });
    
    setImages(updatedImages);
    setProcessedImages(results);
    setShowProcessingDialog(false);
    
    // Show completion dialog instead of immediately going to results
    setShowCompletionDialog(true);
  };

  const handleBatchProcessingComplete = (results: any[]) => {
    console.log('🎉 Batch processing completed successfully:', results);
    
    // Convert batch results to ProcessedImage format
    const processedResults: ProcessedImage[] = results.map(result => ({
      id: Math.random().toString(36).substr(2, 9),
      file: images.find(img => img.file.name === result.image_name)?.file || new File([], result.image_name),
      url: images.find(img => img.file.name === result.image_name)?.url || '',
      objects: result.success ? [{
        type: result.object_type || selectedObjectType,
        count: result.predicted_count || 0
      }] : [],
      resultId: result.result_id,
      processingTime: result.processing_time,
      totalSegments: 0,
      error: result.success ? undefined : result.error
    }));
    
    // Update images with results
    const updatedImages = images.map(img => {
      const result = results.find(r => r.image_name === img.file.name);
      if (result && result.success) {
        return {
          ...img,
          elementCount: result.predicted_count || 0,
          resultId: result.result_id,
          processingError: undefined
        };
      } else if (result && !result.success) {
        return {
          ...img,
          processingError: result.error
        };
      }
      return img;
    });
    
    setImages(updatedImages);
    setProcessedImages(processedResults);
    setShowBatchProcessingDialog(false);
    
    // Show completion dialog
    setShowCompletionDialog(true);
    
    console.log(`✅ Analyzed ${results.length} images successfully`);
  };

  const handleProcessingError = (error: string) => {
    console.error('❌ Processing failed:', error);
    setShowProcessingDialog(false);
    
    // Show error dialog
    showErrorDialog(
      '❌ Processing Failed',
      'Failed to process images. Please try again.',
      error,
      undefined,
      true
    );
  };

  // Completion dialog handlers
  const handleViewResultsFromCompletion = () => {
    setShowCompletionDialog(false);
    setShowResults(true);
  };

  const handleCloseCompletion = () => {
    setShowCompletionDialog(false);
  };

  const totalElements = images.reduce((sum, img) => sum + (img.elementCount || 0), 0);
  
  const handleViewResults = () => {
    setShowResults(true);
  };
  


  if (showHistory) {
    return <ImageHistory onBack={() => setShowHistory(false)} />;
  }

  if (showResults && processedImages.length > 0) {
    return (
      <ResultsDashboard
        images={processedImages}
        onBack={() => setShowResults(false)}
        onViewHistory={() => {
          setShowResults(false);
          setShowHistory(true);
        }}
      />
    );
  }

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex-1"></div>
          <div className="flex-1 text-center">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AI Object Counter
            </h1>
            <div className="flex items-center justify-center gap-2 mt-1">
              <Cpu className="h-4 w-4 text-blue-500" />
              <span className="text-sm text-muted-foreground">with Real-time Performance Monitoring</span>
              <Activity className="h-4 w-4 text-purple-500" />
            </div>
          </div>
          <div className="flex-1 flex justify-end">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setShowHistory(true)}
              className="flex items-center gap-2"
            >
              <History className="h-4 w-4" />
              View History
            </Button>
          </div>
        </div>
        <p className="text-muted-foreground text-center mt-2">
          Upload images and watch AI automatically detect and count all objects with live performance metrics
        </p>
        
        {/* API Status Indicator */}
        {apiStatus === 'checking' && (
          <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent" />
            Connecting to AI backend...
          </div>
        )}
        
        {apiStatus === 'error' && (
          <Alert className="max-w-md mx-auto">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {apiError || 'Unable to connect to backend. Make sure the Flask server is running on port 5000.'}
            </AlertDescription>
          </Alert>
        )}
        
        {apiStatus === 'healthy' && (
          <div className="flex items-center justify-center gap-3 text-sm">
            <div className="flex items-center gap-2 text-green-600">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span>AI Pipeline Ready</span>
            </div>
            <div className="w-1 h-4 bg-gray-300 rounded-full"></div>
            <div className="flex items-center gap-2 text-blue-600">
              <Cpu className="h-3 w-3" />
              <span>Performance Monitor Active</span>
            </div>
            <div className="w-1 h-4 bg-gray-300 rounded-full"></div>
            <div className="flex items-center gap-2 text-purple-600">
              <Zap className="h-3 w-3" />
              <span>GPU Acceleration</span>
            </div>
          </div>
        )}
      </div>

      {/* Upload Area */}
      <Card className="relative">
        <CardContent className="p-0">
          <div
            className={`
              border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer
              ${isDragOver 
                ? 'border-primary bg-primary/5' 
                : 'border-border hover:border-primary/50 hover:bg-accent/50'
              }
            `}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h3 className="mb-2">Drop images here or click to upload</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Support for JPG, PNG, WEBP and other image formats
            </p>
            <Button variant="outline" type="button">
              Choose Files
            </Button>
            <input
              id="file-input"
              type="file"
              multiple
              accept="image/*"
              className="hidden"
              onChange={handleFileInput}
            />
          </div>
        </CardContent>
      </Card>

      {/* Uploaded Images */}
      {images.length > 0 && (
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3>Uploaded Images ({images.length})</h3>
              <Badge variant="secondary">
                Total Elements: {totalElements}
              </Badge>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {images.map((image) => (
                <div key={image.id} className="relative group">
                  <div className="aspect-square rounded-lg overflow-hidden bg-muted">
                    <img
                      src={image.url}
                      alt={image.file.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  
                  {/* Image overlay */}
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors rounded-lg" />
                  
                  {/* Remove button */}
                  <Button
                    variant="destructive"
                    size="sm"
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6 p-0"
                    onClick={() => removeImage(image.id)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                  
                  {/* Element count badge */}
                  {image.elementCount !== undefined && (
                    <Badge className="absolute bottom-2 left-2">
                      {image.elementCount} elements
                    </Badge>
                  )}
                  
                  {/* Processing indicator */}
                  {image.isProcessing && (
                    <div className="absolute inset-0 bg-black/50 rounded-lg flex items-center justify-center">
                      <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent" />
                    </div>
                  )}
                  
                  {/* Error indicator with actions */}
                  {image.processingError && (
                    <div className="absolute inset-0 bg-red-500/20 rounded-lg border-2 border-red-500">
                      <div className="absolute top-2 left-2">
                        <Badge variant="destructive" className="text-xs">
                          Failed
                        </Badge>
                      </div>
                      
                      {/* Error Action Buttons */}
                      <div className="absolute bottom-2 right-2 flex gap-1">
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 w-7 p-0 bg-white hover:bg-gray-100"
                          onClick={(e: React.MouseEvent) => {
                            e.stopPropagation();
                            removeImage(image.id);
                          }}
                          title="Delete image"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 w-7 p-0 bg-white hover:bg-gray-100"
                          onClick={(e: React.MouseEvent) => {
                            e.stopPropagation();
                            retryImageProcessing(image.id);
                          }}
                          title="Retry analysis"
                        >
                          <RefreshCw className="h-3 w-3" />
                        </Button>
                      </div>
                      
                      {/* Click to show error details */}
                      <div 
                        className="absolute inset-0 cursor-pointer flex items-center justify-center"
                        onClick={() => {
                          const isGPUError = image.processingError!.toLowerCase().includes('memory') || 
                                           image.processingError!.toLowerCase().includes('cuda') || 
                                           image.processingError!.toLowerCase().includes('allocator');
                          showErrorDialog(
                            isGPUError ? '🚀 GPU Memory Error' : '❌ Processing Error',
                            isGPUError 
                              ? 'Not enough GPU memory to process this image. Try removing other images or the system will fall back to CPU.'
                              : `Failed to analyze "${image.file.name}". You can delete it or try again.`,
                            image.processingError,
                            image.id,
                            true
                          );
                        }}
                      >
                        <div className="text-center text-white">
                          <AlertCircle className="h-6 w-6 mx-auto mb-1" />
                          <p className="text-xs font-medium">Click for details</p>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* File name */}
                  <p className="text-xs text-muted-foreground mt-1 truncate">
                    {image.file.name}
                  </p>
                  
                  {/* Error message */}
                  {image.processingError && (
                    <p className="text-xs text-red-500 mt-1">
                      {image.processingError}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Object Type Selection */}
      <Card>
        <CardContent className="p-6 space-y-4">
          <div className="space-y-2">
            <label htmlFor="object-type">Select Object Type to Detect</label>
            <Select value={selectedObjectType} onValueChange={setSelectedObjectType}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Choose an object type to detect and count" />
              </SelectTrigger>
              <SelectContent>
                {objectTypes.map((objectType) => (
                  <SelectItem key={objectType} value={objectType}>
                    <span className="capitalize">{objectType}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              The AI will detect and count only the selected object type in your images. This makes processing much faster and more accurate.
            </p>
          </div>
          
        </CardContent>
      </Card>

      {/* Action Button */}
      <div className="flex justify-center">
        <Button
          onClick={processImages}
          disabled={images.length === 0 || apiStatus !== 'healthy'}
          className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold"
          size="lg"
        >
          <Zap className="h-5 w-5 mr-2" />
          🚀 Start AI Analysis ({images.length} {images.length === 1 ? 'image' : 'images'})
          <Activity className="h-4 w-4 ml-2" />
        </Button>
      </div>

      {/* Results Summary */}
      {totalElements > 0 && (
        <Card>
          <CardContent className="p-6 text-center space-y-4">
            <ImageIcon className="h-12 w-12 mx-auto mb-4 text-primary" />
            <h3 className="mb-2">Analysis Complete!</h3>
            <p className="text-muted-foreground">
              Detected a total of <strong>{totalElements} objects</strong> across {images.length} {images.length === 1 ? 'image' : 'images'}
            </p>
            
            {/* Show processing errors with actions */}
            {images.some(img => img.processingError) && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-red-500" />
                    <span className="text-sm font-medium text-red-700">
                      {images.filter(img => img.processingError).length} image(s) failed to analyze
                    </span>
                  </div>
                  
                  {/* Bulk Actions for Failed Images */}
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const failedImages = images.filter(img => img.processingError);
                        failedImages.forEach(img => removeImage(img.id));
                      }}
                      className="text-red-600 border-red-300 hover:bg-red-50"
                    >
                      <Trash2 className="h-3 w-3 mr-1" />
                      Remove All Failed
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={async () => {
                        const failedImages = images.filter(img => img.processingError);
                        for (const img of failedImages) {
                          await retryImageProcessing(img.id);
                        }
                      }}
                      className="text-blue-600 border-blue-300 hover:bg-blue-50"
                    >
                      <RefreshCw className="h-3 w-3 mr-1" />
                      Retry All Failed
                    </Button>
                  </div>
                </div>
                
                <p className="text-xs text-red-600">
                  Click on failed images for details, or use the buttons above for bulk actions.
                </p>
              </div>
            )}
            <Button onClick={handleViewResults} className="mt-4 flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Open Results Dashboard
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Enhanced Error Dialog */}
      <Dialog open={errorDialog.isOpen} onOpenChange={closeErrorDialog}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {errorDialog.title}
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              {errorDialog.message}
            </p>
            
            {errorDialog.details && (
              <div className="bg-muted p-3 rounded-md">
                <p className="text-xs font-mono text-muted-foreground">
                  {errorDialog.details}
                </p>
              </div>
            )}
          </div>
          
          <DialogFooter className="gap-2">
            {errorDialog.canRetry && errorDialog.imageId && (
              <>
                <Button
                  variant="outline"
                  onClick={() => removeImage(errorDialog.imageId!)}
                  className="flex items-center gap-2"
                >
                  <Trash2 className="h-4 w-4" />
                  Delete Image
                </Button>
                <Button
                  onClick={() => retryImageProcessing(errorDialog.imageId!)}
                  className="flex items-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" />
                  Retry
                </Button>
              </>
            )}
            <Button variant="secondary" onClick={closeErrorDialog}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Simple Processing Dialog */}
      <SimpleProcessingDialog
        isOpen={showProcessingDialog}
        onClose={() => setShowProcessingDialog(false)}
        totalImages={images.length}
        imageFiles={images.map(img => img.file)}
        selectedObjectType={selectedObjectType}
        onProcessingComplete={handleProcessingComplete}
        onProcessingError={handleProcessingError}
      />

      {/* Batch Processing Dialog */}
      <BatchProcessingDialog
        isOpen={showBatchProcessingDialog}
        onClose={() => setShowBatchProcessingDialog(false)}
        imageFiles={images.map(img => img.file)}
        objectType={selectedObjectType}
        description={`Batch processing ${selectedObjectType} objects`}
        autoDetect={false}
        onProcessingComplete={handleBatchProcessingComplete}
        onProcessingError={handleProcessingError}
      />

      {/* Completion Dialog */}
      <CompletionDialog
        isOpen={showCompletionDialog}
        onClose={handleCloseCompletion}
        onViewResults={handleViewResultsFromCompletion}
        totalImages={processedImages.length}
        successfulImages={processedImages.filter(img => !img.error).length}
        failedImages={processedImages.filter(img => img.error).length}
        totalObjects={processedImages
          .filter(img => !img.error)
          .reduce((sum, img) => sum + img.objects.reduce((objSum, obj) => objSum + obj.count, 0), 0)
        }
      />
    </div>
  );
}
