import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Button } from './ui/button';

interface BatchProcessingDialogProps {
  isOpen: boolean;
  onClose: () => void;
  imageFiles: File[];
  objectType: string;
  description?: string;
  autoDetect?: boolean;
  onProcessingComplete: (results: any[]) => void;
  onProcessingError: (error: string) => void;
}

export function BatchProcessingDialog({
  isOpen,
  onClose,
  imageFiles,
  objectType,
  description,
  autoDetect = false,
  onProcessingComplete,
  onProcessingError
}: BatchProcessingDialogProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Batch Processing</DialogTitle>
        </DialogHeader>
        <div className="p-4">
          <p>Processing {imageFiles.length} images for {objectType}...</p>
          <p>This feature is coming soon!</p>
        </div>
        <div className="flex justify-end">
          <Button onClick={onClose}>Close</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}








