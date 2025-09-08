import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from './ui/dialog';
import { Button } from './ui/button';
import { CheckCircle2, BarChart3, Camera, ArrowRight, X } from 'lucide-react';
import { Badge } from './ui/badge';

interface CompletionDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onViewResults: () => void;
  totalImages: number;
  successfulImages: number;
  failedImages: number;
  totalObjects: number;
}

export function CompletionDialog({
  isOpen,
  onClose,
  onViewResults,
  totalImages,
  successfulImages,
  failedImages,
  totalObjects
}: CompletionDialogProps) {
  const handleViewResults = () => {
    onViewResults();
    onClose();
  };

  const handleStayHere = () => {
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-md mx-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-center justify-center">
            <CheckCircle2 className="h-6 w-6 text-green-500" />
            <span>Analysis Complete!</span>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Success Animation */}
          <div className="text-center">
            <div className="relative inline-block">
              <div className="absolute inset-0 bg-green-100 rounded-full animate-ping opacity-75"></div>
              <CheckCircle2 className="relative h-16 w-16 text-green-500 mx-auto" />
            </div>
          </div>

          {/* Summary Stats */}
          <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-4 space-y-3">
            <h3 className="font-medium text-center text-gray-800 mb-3">Processing Summary</h3>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 mb-1">
                  <Camera className="h-4 w-4 text-blue-500" />
                  <span className="font-medium">Images</span>
                </div>
                <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                  {totalImages} processed
                </Badge>
              </div>
              
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 mb-1">
                  <BarChart3 className="h-4 w-4 text-green-500" />
                  <span className="font-medium">Objects</span>
                </div>
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                  {totalObjects} detected
                </Badge>
              </div>
            </div>

            {/* Status Breakdown */}
            <div className="flex items-center justify-center gap-4 pt-2 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>{successfulImages} successful</span>
              </div>
              {failedImages > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                  <span>{failedImages} failed</span>
                </div>
              )}
            </div>
          </div>

          {/* Call to Action */}
          <div className="text-center space-y-2">
            <p className="text-gray-600">
              Would you like to view the detailed results dashboard?
            </p>
            <p className="text-sm text-gray-500">
              Explore performance metrics, export data, and provide feedback
            </p>
          </div>
        </div>

        <DialogFooter className="flex gap-2">
          <Button
            variant="outline"
            onClick={handleStayHere}
            className="flex-1"
          >
            <X className="h-4 w-4 mr-2" />
            Stay Here
          </Button>
          
          <Button
            onClick={handleViewResults}
            className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
          >
            <BarChart3 className="h-4 w-4 mr-2" />
            View Dashboard
            <ArrowRight className="h-4 w-4 ml-1" />
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}


