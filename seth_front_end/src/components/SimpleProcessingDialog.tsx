import { useState, useEffect } from 'react';
import { Dialog, DialogContent } from './ui/dialog';
import { Button } from './ui/button';
import { X, Loader2, Camera, CheckCircle, AlertCircle } from 'lucide-react';
import api from '../services/api';
import { ProcessedImage } from './ResultsDashboard';

interface SimpleProcessingDialogProps {
  isOpen: boolean;
  onClose: () => void;
  imageFiles: File[];
  selectedObjectType: string;
  totalImages: number;
  onProcessingComplete: (results: ProcessedImage[]) => void;
  onProcessingError: (error: string) => void;
}

export function SimpleProcessingDialog({
  isOpen,
  onClose,
  imageFiles,
  selectedObjectType,
  totalImages,
  onProcessingComplete,
  onProcessingError
}: SimpleProcessingDialogProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [processedImages, setProcessedImages] = useState<ProcessedImage[]>([]);
  const [processingStage, setProcessingStage] = useState('');
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Reset state when dialog opens
  useEffect(() => {
    if (isOpen) {
      setIsProcessing(false);
      setCurrentImageIndex(0);
      setProcessedImages([]);
      setProcessingStage('');
      setHasError(false);
      setErrorMessage('');
    }
  }, [isOpen]);

  const startProcessing = async () => {
    if (imageFiles.length === 0) return;

    setIsProcessing(true);
    setProcessingStage('Initializing AI models...');
    const results: ProcessedImage[] = [];

    try {
      for (let i = 0; i < imageFiles.length; i++) {
        const imageFile = imageFiles[i];
        setCurrentImageIndex(i);
        setProcessingStage(`Processing image ${i + 1} of ${imageFiles.length}...`);

        try {
          const result = await api.countObjects(
            imageFile,
            selectedObjectType,
            `Count ${selectedObjectType} objects in this image`
          );

          const processedImage: ProcessedImage = {
            id: Math.random().toString(36).substr(2, 9),
            file: imageFile,
            url: URL.createObjectURL(imageFile),
            objects: [{
              type: result.object_type,
              count: result.predicted_count
            }],
            resultId: result.result_id,
            processingTime: result.processing_time,
            totalSegments: result.total_segments
          };

          results.push(processedImage);
          setProcessedImages([...results]);

          // Small delay to show progress
          await new Promise(resolve => setTimeout(resolve, 200));
        } catch (error) {
          console.error(`Error processing image ${i + 1}:`, error);
          
          // Add failed image to results
          const failedImage: ProcessedImage = {
            id: Math.random().toString(36).substr(2, 9),
            file: imageFile,
            url: URL.createObjectURL(imageFile),
            objects: [],
            error: error instanceof Error ? error.message : 'Processing failed'
          };
          
          results.push(failedImage);
          setProcessedImages([...results]);
        }
      }

      setProcessingStage('Analysis complete!');
      
      // Short delay before showing completion
      setTimeout(() => {
        setIsProcessing(false);
        onProcessingComplete(results);
      }, 1000);

    } catch (error) {
      console.error('Processing failed:', error);
      setHasError(true);
      setErrorMessage(error instanceof Error ? error.message : 'Processing failed');
      setIsProcessing(false);
      onProcessingError(error instanceof Error ? error.message : 'Processing failed');
    }
  };

  const handleCancel = () => {
    setIsProcessing(false);
    onClose();
  };

  const getProgressText = () => {
    if (hasError) return 'Processing failed';
    if (!isProcessing && processedImages.length > 0) return 'Analysis complete!';
    if (processingStage) return processingStage;
    return 'Ready to process';
  };

  const getProgressPercentage = () => {
    if (totalImages === 0) return 0;
    return Math.round((processedImages.length / totalImages) * 100);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-md mx-auto">
        <div className="flex flex-col items-center text-center space-y-6 py-8">
          {/* Header */}
          <div className="flex items-center justify-between w-full">
            <h2 className="text-xl font-semibold">Processing Images</h2>
            {!isProcessing && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>

          {/* Main Status */}
          <div className="space-y-4">
            {/* Spinner or Status Icon */}
            <div className="flex justify-center">
              {hasError ? (
                <AlertCircle className="h-16 w-16 text-red-500" />
              ) : !isProcessing && processedImages.length > 0 ? (
                <CheckCircle className="h-16 w-16 text-green-500" />
              ) : isProcessing ? (
                <Loader2 className="h-16 w-16 text-blue-500 animate-spin" />
              ) : (
                <Camera className="h-16 w-16 text-gray-400" />
              )}
            </div>

            {/* Status Text */}
            <div className="space-y-2">
              <p className="text-lg font-medium">{getProgressText()}</p>
              
              {isProcessing && (
                <div className="space-y-2">
                  <p className="text-sm text-gray-600">
                    {processedImages.length} of {totalImages} images processed ({getProgressPercentage()}%)
                  </p>
                  
                  {/* Simple Progress Bar */}
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${getProgressPercentage()}%` }}
                    />
                  </div>
                </div>
              )}

              {hasError && (
                <div className="p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
                  {errorMessage}
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="space-y-2 w-full">
            {!isProcessing && processedImages.length === 0 && !hasError && (
              <Button onClick={startProcessing} className="w-full" size="lg">
                <Camera className="h-4 w-4 mr-2" />
                Start AI Analysis
              </Button>
            )}

            {isProcessing && (
              <Button onClick={handleCancel} variant="outline" className="w-full">
                Cancel Processing
              </Button>
            )}

            {hasError && (
              <div className="flex gap-2 w-full">
                <Button onClick={startProcessing} className="flex-1">
                  Retry
                </Button>
                <Button onClick={onClose} variant="outline" className="flex-1">
                  Close
                </Button>
              </div>
            )}
          </div>

          {/* Image Counter */}
          <div className="text-sm text-gray-500">
            Processing {totalImages} image{totalImages === 1 ? '' : 's'}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
