import React, { useState, useEffect } from 'react';
import { ArrowLeft, MessageSquare, X, Plus, Check, Clock, Layers, History } from 'lucide-react';
import api, { ApiObjectType } from '../services/api';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';

export interface ObjectCount {
  type: string;
  count: number;
}

export interface ProcessedImage {
  id: string;
  file: File;
  url: string;
  objects: ObjectCount[];
  resultId?: string;
  processingTime?: number;
  totalSegments?: number;
}

interface CorrectionTag {
  id: string;
  type: string;
  count: number;
}

interface ResultsProps {
  images: ProcessedImage[];
  onBack: () => void;
  onViewHistory?: () => void;
}

export function Results({ images, onBack, onViewHistory }: ResultsProps) {
  const [selectedImage, setSelectedImage] = useState<ProcessedImage | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [objectType, setObjectType] = useState('');
  const [objectCount, setObjectCount] = useState('');
  const [corrections, setCorrections] = useState<CorrectionTag[]>([]);
  const [feedbackSaved, setFeedbackSaved] = useState<Set<string>>(new Set());
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string>('');
  const [objectTypes, setObjectTypes] = useState<ApiObjectType[]>([]);

  // Load object types from API
  useEffect(() => {
    const loadObjectTypes = async () => {
      try {
        const types = await api.getObjectTypes();
        setObjectTypes(types);
        
        // Set default object type to first available type
        if (types.length > 0 && !objectType) {
          setObjectType(types[0].name);
        }
      } catch (error) {
        console.error('Failed to load object types:', error);
      }
    };

    loadObjectTypes();
  }, []);

  // Calculate total counts across all images
  const totalCounts = images.reduce((acc, image) => {
    image.objects.forEach(obj => {
      const existing = acc.find(item => item.type === obj.type);
      if (existing) {
        existing.count += obj.count;
      } else {
        acc.push({ type: obj.type, count: obj.count });
      }
    });
    return acc;
  }, [] as ObjectCount[]);

  const openFeedbackDialog = (image: ProcessedImage) => {
    setSelectedImage(image);
    setCorrections([]);
    setObjectType(objectTypes.length > 0 ? objectTypes[0].name : '');
    setObjectCount('');
    setFeedbackError('');
    setIsDialogOpen(true);
  };

  const addCorrection = () => {
    if (objectType.trim() && objectCount.trim() && !isNaN(Number(objectCount))) {
      const newCorrection: CorrectionTag = {
        id: Math.random().toString(36).substr(2, 9),
        type: objectType.trim(),
        count: parseInt(objectCount)
      };
      setCorrections(prev => [...prev, newCorrection]);
      setObjectType('');
      setObjectCount('');
    }
  };

  const removeCorrection = (id: string) => {
    setCorrections(prev => prev.filter(correction => correction.id !== id));
  };

  const submitFeedback = async () => {
    if (!selectedImage || !selectedImage.resultId) {
      setFeedbackError('No result ID available for correction');
      return;
    }

    setSubmittingFeedback(true);
    setFeedbackError('');

    try {
      if (corrections.length > 0) {
        // User has provided corrections - submit the first one
        const mainCorrection = corrections[0];
        
        console.log(`📝 Submitting correction for result ${selectedImage.resultId}: ${mainCorrection.count}`);
        
        const response = await api.correctPrediction(
          selectedImage.resultId,
          mainCorrection.count
        );
        
        console.log('✅ Correction submitted successfully:', response);
        
        // Mark as corrected and close dialog
        setFeedbackSaved(prev => new Set(prev).add(selectedImage.id));
        setIsDialogOpen(false);
        setSelectedImage(null);
        setCorrections([]);
        
        // Show success message
        console.log(`🎯 Feedback saved! AI predicted: ${selectedImage.objects.reduce((sum, obj) => sum + obj.count, 0)}, User corrected: ${mainCorrection.count}`);
        
      } else {
        // User wants to confirm AI prediction is correct (no corrections needed)
        const totalAIPredicted = selectedImage.objects.reduce((sum, obj) => sum + obj.count, 0);
        
        console.log(`✅ Confirming AI prediction is correct for result ${selectedImage.resultId}: ${totalAIPredicted}`);
        
        const response = await api.correctPrediction(
          selectedImage.resultId,
          totalAIPredicted // Confirm the AI prediction was correct
        );
        
        console.log('✅ Confirmation submitted successfully:', response);
        
        // Mark as reviewed and close dialog
        setFeedbackSaved(prev => new Set(prev).add(selectedImage.id));
        setIsDialogOpen(false);
        setSelectedImage(null);
        
        console.log(`🎯 AI prediction confirmed as correct: ${totalAIPredicted} objects`);
      }
      
    } catch (error) {
      console.error('❌ Failed to submit feedback:', error);
      setFeedbackError(error instanceof Error ? error.message : 'Failed to submit feedback. Please try again.');
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const totalObjects = totalCounts.reduce((sum, obj) => sum + obj.count, 0);

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={onBack} className="flex items-center gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Upload
            </Button>
            <div>
              <h1>AI Detection Results</h1>
              <p className="text-muted-foreground">
                Automatic object detection completed for {images.length} {images.length === 1 ? 'image' : 'images'}
              </p>
            </div>
          </div>
          {onViewHistory && (
            <Button variant="outline" onClick={onViewHistory} className="flex items-center gap-2">
              <History className="h-4 w-4" />
              View History
            </Button>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Left side - Images Grid (3/4 width) */}
          <div className="lg:col-span-3">
            <Card>
              <CardHeader>
                <CardTitle>AI Detection Results</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Automatic object detection and counting completed
                </p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {images.map((image, index) => (
                    <div key={image.id} className="space-y-4">
                      {/* Image */}
                      <div className="aspect-square rounded-lg overflow-hidden bg-muted">
                        <img
                          src={image.url}
                          alt={`Image ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      
                                              {/* Image Info */}
                        <div className="space-y-3">
                          <h4>Image {index + 1}</h4>
                          
                          {/* Processing Info */}
                          {(image.processingTime || image.totalSegments) && (
                            <div className="flex gap-4 text-xs text-muted-foreground">
                              {image.processingTime && (
                                <div className="flex items-center gap-1">
                                  <Clock className="h-3 w-3" />
                                  {image.processingTime}s
                                </div>
                              )}
                              {image.totalSegments && (
                                <div className="flex items-center gap-1">
                                  <Layers className="h-3 w-3" />
                                  {image.totalSegments} segments
                                </div>
                              )}
                            </div>
                          )}
                          
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <p className="text-sm font-medium">Detected Objects</p>
                              <Badge variant="outline" className="text-xs">
                                {image.objects.reduce((sum, obj) => sum + obj.count, 0)} total
                              </Badge>
                            </div>
                            {image.objects.length > 0 ? (
                              <div className="space-y-2">
                                {image.objects.map((obj, objIndex) => (
                                  <div key={objIndex} className="flex justify-between items-center p-2 bg-muted/50 rounded-md">
                                    <div className="flex items-center gap-2">
                                      <div className="w-2 h-2 bg-primary rounded-full"></div>
                                      <span className="capitalize text-sm font-medium">{obj.type}</span>
                                    </div>
                                    <Badge variant="secondary" className="font-semibold">{obj.count}</Badge>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-center py-4 text-muted-foreground text-sm">
                                No objects detected
                              </div>
                            )}
                          </div>
                        
                        {/* Feedback Button */}
                        <Button 
                          variant="outline" 
                          className="w-full"
                          onClick={() => openFeedbackDialog(image)}
                        >
                          <MessageSquare className="h-4 w-4 mr-2" />
                          {feedbackSaved.has(image.id) ? 'Feedback Saved' : 'Feedback'}
                          {feedbackSaved.has(image.id) && <Check className="h-4 w-4 ml-2" />}
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right side - Total Summary (1/4 width) */}
          <div className="lg:col-span-1">
            <Card className="sticky top-8">
              <CardHeader>
                <CardTitle>Detection Summary</CardTitle>
                <p className="text-xs text-muted-foreground">
                  Across {images.length} {images.length === 1 ? 'image' : 'images'}
                </p>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center p-4 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg border">
                  <div className="text-3xl font-bold text-primary">{totalObjects}</div>
                  <div className="text-sm text-muted-foreground font-medium">Total Objects Detected</div>
                </div>
                
                <Separator />
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium">Object Types Found:</p>
                    <Badge variant="secondary">{totalCounts.length} types</Badge>
                  </div>
                  {totalCounts.length > 0 ? (
                    <div className="space-y-3">
                      {totalCounts.map((obj, index) => (
                        <div key={index} className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="capitalize text-sm font-medium">{obj.type}</span>
                            <Badge variant="outline" className="font-semibold">{obj.count}</Badge>
                          </div>
                          <div className="w-full bg-muted rounded-full h-1.5">
                            <div 
                              className="bg-primary h-1.5 rounded-full transition-all duration-300"
                              style={{ 
                                width: `${(obj.count / Math.max(...totalCounts.map(t => t.count))) * 100}%` 
                              }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-4 text-muted-foreground text-xs">
                      No objects detected across all images
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Feedback Dialog */}
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogContent className="max-w-5xl w-[95vw] max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Provide Feedback</DialogTitle>
            </DialogHeader>
            
            {selectedImage && (
              <div className="space-y-6">
                {/* Enlarged Image */}
                <div className="w-full h-64 rounded-lg overflow-hidden bg-muted">
                  <img
                    src={selectedImage.url}
                    alt="Feedback Image"
                    className="w-full h-full object-contain"
                  />
                </div>
                
                {/* Current Details */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h4>AI Detection Results:</h4>
                    <Badge variant="outline">
                      {selectedImage.objects.reduce((sum, obj) => sum + obj.count, 0)} total objects
                    </Badge>
                  </div>
                  {selectedImage.objects.length > 0 ? (
                    <div className="grid grid-cols-2 gap-2">
                      {selectedImage.objects.map((obj, index) => (
                        <div key={index} className="flex justify-between items-center p-3 bg-muted rounded-md border">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-primary rounded-full"></div>
                            <span className="capitalize font-medium">{obj.type}</span>
                          </div>
                          <Badge variant="secondary">{obj.count}</Badge>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-4 text-muted-foreground">
                      No objects detected in this image
                    </div>
                  )}
                </div>
                
                <Separator />
                
                {/* Instructions */}
                <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                  <h4 className="text-sm font-medium text-blue-900 mb-2">📝 Provide Feedback</h4>
                  <div className="text-xs text-blue-700 space-y-1">
                    <p><strong>Option 1:</strong> Click "Submit Feedback" to confirm the AI detection is correct</p>
                    <p><strong>Option 2:</strong> Add a correction below, then submit</p>
                  </div>
                </div>
                
                {/* Correction Form */}
                <div className="space-y-4">
                  <h4>Add Correction (Optional):</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label htmlFor="object-type">Object Type</label>
                      <Select value={objectType} onValueChange={setObjectType}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select object type" />
                        </SelectTrigger>
                        <SelectContent>
                          {objectTypes.map((type) => (
                            <SelectItem key={type.id} value={type.name}>
                              <div className="flex flex-col items-start">
                                <span className="capitalize font-medium">{type.name}</span>
                                <span className="text-xs text-muted-foreground">{type.description}</span>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="object-count">Number of Objects</label>
                      <div className="flex gap-2">
                        <Input
                          id="object-count"
                          type="number"
                          placeholder="0"
                          value={objectCount}
                          onChange={(e) => setObjectCount(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && addCorrection()}
                        />
                        <Button 
                          onClick={addCorrection}
                          disabled={!objectType.trim() || !objectCount.trim() || isNaN(Number(objectCount))}
                        >
                          <Plus className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                  
                  {/* Correction Tags */}
                  {corrections.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Your Corrections:</p>
                      <div className="flex flex-wrap gap-2">
                        {corrections.map((correction) => (
                          <Badge key={correction.id} variant="outline" className="flex items-center gap-2 px-3 py-1">
                            {correction.type}: {correction.count}
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-4 w-4 p-0 hover:bg-destructive hover:text-destructive-foreground"
                              onClick={() => removeCorrection(correction.id)}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Error Display */}
                {feedbackError && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-md">
                    <p className="text-sm text-red-700">{feedbackError}</p>
                  </div>
                )}
                
                {/* Submit Button */}
                <div className="flex justify-end gap-3">
                  <Button variant="outline" onClick={() => setIsDialogOpen(false)} disabled={submittingFeedback}>
                    Cancel
                  </Button>
                  <Button onClick={submitFeedback} disabled={submittingFeedback}>
                    {submittingFeedback ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2" />
                        Submitting...
                      </>
                    ) : (
                      <>
                        <Check className="h-4 w-4 mr-2" />
                        {corrections.length > 0 
                          ? `Submit Correction (${corrections[0].count} objects)`
                          : 'Confirm AI is Correct'
                        }
                      </>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
