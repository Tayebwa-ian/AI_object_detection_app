import React, { useState, useEffect } from 'react';
import { 
  X, 
  Download, 
  Share2, 
  BarChart3, 
  History, 
  Grid3X3, 
  Camera,
  Clock,
  Target,
  TrendingUp,
  FileText,
  Layers,
  Eye,
  Plus,
  Check,
  MessageSquare,
  ExternalLink,
  Maximize2,
  Filter
} from 'lucide-react';
import api, { ApiObjectType } from '../services/api';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { F1ScoreDisplay } from './F1ScoreDisplay';

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
  error?: string;
}

interface CorrectionTag {
  id: string;
  type: string;
  count: number;
}

interface ResultsModalProps {
  isOpen: boolean;
  onClose: () => void;
  images: ProcessedImage[];
  onViewHistory?: () => void;
}

export function ResultsModal({ isOpen, onClose, images, onViewHistory }: ResultsModalProps) {
  const [activeTab, setActiveTab] = useState('results');
  const [selectedImage, setSelectedImage] = useState<ProcessedImage | null>(null);
  const [isDetailDialogOpen, setIsDetailDialogOpen] = useState(false);
  const [objectType, setObjectType] = useState('');
  const [objectCount, setObjectCount] = useState('');
  const [corrections, setCorrections] = useState<CorrectionTag[]>([]);
  const [feedbackSaved, setFeedbackSaved] = useState<Set<string>>(new Set());
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string>('');
  const [objectTypes, setObjectTypes] = useState<ApiObjectType[]>([]);
  const [filterType, setFilterType] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('newest');

  // Load object types from API
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

  // Calculate performance statistics
  const calculateStats = () => {
    const successfulImages = images.filter(img => !img.error);
    const failedImages = images.filter(img => img.error);
    
    const totalObjects = successfulImages.reduce((sum, img) => 
      sum + img.objects.reduce((objSum, obj) => objSum + obj.count, 0), 0
    );
    
    const avgProcessingTime = successfulImages.length > 0 
      ? successfulImages.reduce((sum, img) => sum + (img.processingTime || 0), 0) / successfulImages.length 
      : 0;
    
    const totalSegments = successfulImages.reduce((sum, img) => sum + (img.totalSegments || 0), 0);
    
    // Object type distribution
    const objectTypeCounts: { [key: string]: number } = {};
    successfulImages.forEach(img => {
      img.objects.forEach(obj => {
        objectTypeCounts[obj.type] = (objectTypeCounts[obj.type] || 0) + obj.count;
      });
    });

    return {
      totalImages: images.length,
      successfulImages: successfulImages.length,
      failedImages: failedImages.length,
      totalObjects,
      avgProcessingTime,
      totalSegments,
      objectTypeCounts,
      successRate: images.length > 0 ? (successfulImages.length / images.length) * 100 : 0
    };
  };

  // Export functionality
  const exportResults = (format: 'json' | 'csv') => {
    const stats = calculateStats();
    const exportData = {
      summary: stats,
      images: images.map(img => ({
        id: img.id,
        filename: img.file.name,
        objects: img.objects,
        processingTime: img.processingTime,
        totalSegments: img.totalSegments,
        error: img.error,
        totalObjectCount: img.objects.reduce((sum, obj) => sum + obj.count, 0)
      }))
    };

    if (format === 'json') {
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ai-analysis-results-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } else if (format === 'csv') {
      const csvRows = [
        'Filename,Object Type,Count,Processing Time (s),Total Segments,Status',
        ...images.flatMap(img => 
          img.error 
            ? [`${img.file.name},ERROR,0,0,0,Failed: ${img.error}`]
            : img.objects.map(obj => 
                `${img.file.name},${obj.type},${obj.count},${img.processingTime || 0},${img.totalSegments || 0},Success`
              )
        )
      ];
      
      const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ai-analysis-results-${new Date().toISOString().split('T')[0]}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  // Filter and sort images
  const getFilteredAndSortedImages = () => {
    let filtered = [...images];
    
    // Apply filter
    if (filterType !== 'all') {
      if (filterType === 'success') {
        filtered = filtered.filter(img => !img.error);
      } else if (filterType === 'error') {
        filtered = filtered.filter(img => img.error);
      } else {
        filtered = filtered.filter(img => 
          img.objects.some(obj => obj.type === filterType)
        );
      }
    }
    
    // Apply sort
    switch (sortBy) {
      case 'newest':
        // Maintain original order (newest first)
        break;
      case 'oldest':
        filtered.reverse();
        break;
      case 'most-objects':
        filtered.sort((a, b) => {
          const aCount = a.objects.reduce((sum, obj) => sum + obj.count, 0);
          const bCount = b.objects.reduce((sum, obj) => sum + obj.count, 0);
          return bCount - aCount;
        });
        break;
      case 'least-objects':
        filtered.sort((a, b) => {
          const aCount = a.objects.reduce((sum, obj) => sum + obj.count, 0);
          const bCount = b.objects.reduce((sum, obj) => sum + obj.count, 0);
          return aCount - bCount;
        });
        break;
      case 'fastest':
        filtered.sort((a, b) => (a.processingTime || 0) - (b.processingTime || 0));
        break;
    }
    
    return filtered;
  };

  // Submit feedback for corrections
  const submitFeedback = async () => {
    if (!selectedImage || corrections.length === 0) return;

    setSubmittingFeedback(true);
    setFeedbackError('');

    try {
      for (const correction of corrections) {
        if (selectedImage.resultId) {
          await api.correctPrediction(selectedImage.resultId, correction.count);
        }
      }

      setFeedbackSaved(prev => new Set([...prev, selectedImage.id]));
      setIsDetailDialogOpen(false);
      setCorrections([]);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      setFeedbackError(error instanceof Error ? error.message : 'Failed to submit feedback');
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const addCorrection = () => {
    if (objectType && objectCount) {
      const newCorrection: CorrectionTag = {
        id: Math.random().toString(36).substr(2, 9),
        type: objectType,
        count: parseInt(objectCount)
      };
      setCorrections([...corrections, newCorrection]);
      setObjectType('');
      setObjectCount('');
    }
  };

  const removeCorrection = (id: string) => {
    setCorrections(corrections.filter(c => c.id !== id));
  };

  const openImageDetail = (image: ProcessedImage) => {
    setSelectedImage(image);
    setIsDetailDialogOpen(true);
    setCorrections([]);
    setFeedbackError('');
  };

  const stats = calculateStats();
  const filteredImages = getFilteredAndSortedImages();

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-[95vw] max-h-[95vh] w-full h-full overflow-hidden bg-gradient-to-br from-slate-50 via-blue-50 to-slate-50">
        <DialogHeader className="border-b border-gray-200 pb-4">
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-3 text-2xl">
              <BarChart3 className="h-7 w-7 text-blue-600" />
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                AI Analysis Results
              </span>
              <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-300">
                {images.length} images
              </Badge>
            </DialogTitle>
            
            <div className="flex items-center gap-2">
              {onViewHistory && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    onViewHistory();
                    onClose();
                  }}
                  className="flex items-center gap-2"
                >
                  <History className="h-4 w-4" />
                  View History
                </Button>
              )}
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => exportResults('json')}
                className="flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Export JSON
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => exportResults('csv')}
                className="flex items-center gap-2"
              >
                <FileText className="h-4 w-4" />
                Export CSV
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </DialogHeader>
        
        <div className="flex-1 overflow-hidden">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
            <TabsList className="grid grid-cols-3 w-full max-w-md mx-auto mb-4">
              <TabsTrigger value="results" className="flex items-center gap-2">
                <Grid3X3 className="h-4 w-4" />
                Results
              </TabsTrigger>
              <TabsTrigger value="performance" className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Performance
              </TabsTrigger>
              <TabsTrigger value="history" className="flex items-center gap-2">
                <History className="h-4 w-4" />
                Analysis
              </TabsTrigger>
            </TabsList>
            
            {/* Results Tab */}
            <TabsContent value="results" className="flex-1 overflow-hidden">
              <div className="h-full flex flex-col">
                {/* Filter and Sort Controls */}
                <div className="flex items-center gap-4 mb-4 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
                  <div className="flex items-center gap-2">
                    <Filter className="h-4 w-4 text-gray-600" />
                    <Select value={filterType} onValueChange={setFilterType}>
                      <SelectTrigger className="w-40">
                        <SelectValue placeholder="Filter by type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Images</SelectItem>
                        <SelectItem value="success">Successful Only</SelectItem>
                        <SelectItem value="error">Failed Only</SelectItem>
                        <Separator />
                        {Object.keys(stats.objectTypeCounts).map(type => (
                          <SelectItem key={type} value={type}>
                            {type} ({stats.objectTypeCounts[type]})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Sort:</span>
                    <Select value={sortBy} onValueChange={setSortBy}>
                      <SelectTrigger className="w-40">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="newest">Newest First</SelectItem>
                        <SelectItem value="oldest">Oldest First</SelectItem>
                        <SelectItem value="most-objects">Most Objects</SelectItem>
                        <SelectItem value="least-objects">Least Objects</SelectItem>
                        <SelectItem value="fastest">Fastest Processing</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="ml-auto text-sm text-gray-600">
                    Showing {filteredImages.length} of {images.length} images
                  </div>
                </div>
                
                {/* Results Grid */}
                <div className="flex-1 overflow-y-auto pr-2">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 pb-4">
                    {filteredImages.map((image) => (
                      <Card 
                        key={image.id} 
                        className="group hover:shadow-lg transition-all duration-200 cursor-pointer border-2 hover:border-blue-300"
                        onClick={() => openImageDetail(image)}
                      >
                        <CardContent className="p-4">
                          {/* Image Preview */}
                          <div className="aspect-square rounded-lg bg-gray-100 flex items-center justify-center overflow-hidden mb-3 relative">
                            {image.error ? (
                              <div className="text-red-500 text-center p-4">
                                <X className="h-8 w-8 mx-auto mb-2" />
                                <p className="text-sm">Processing Failed</p>
                              </div>
                            ) : (
                              <>
                                <img 
                                  src={image.url} 
                                  alt="Analyzed" 
                                  className="w-full h-full object-cover"
                                />
                                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                  <Badge className="bg-white/90 text-gray-800">
                                    <Maximize2 className="h-3 w-3 mr-1" />
                                    View
                                  </Badge>
                                </div>
                              </>
                            )}
                          </div>
                          
                          {/* File Info */}
                          <div className="mb-3">
                            <h3 className="font-medium text-sm truncate" title={image.file.name}>
                              {image.file.name}
                            </h3>
                            <p className="text-xs text-gray-500">
                              {(image.file.size / 1024 / 1024).toFixed(1)} MB
                            </p>
                          </div>
                          
                          {/* Results Summary */}
                          {image.error ? (
                            <div className="text-red-600 text-sm p-2 bg-red-50 rounded border border-red-200">
                              <strong>Error:</strong> {image.error}
                            </div>
                          ) : (
                            <>
                              {/* Object Counts */}
                              <div className="space-y-2 mb-3">
                                {image.objects.slice(0, 3).map((obj, idx) => (
                                  <div key={idx} className="flex items-center justify-between text-sm">
                                    <span className="capitalize text-gray-600">{obj.type}</span>
                                    <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                                      {obj.count}
                                    </Badge>
                                  </div>
                                ))}
                                {image.objects.length > 3 && (
                                  <div className="text-xs text-gray-500 text-center">
                                    +{image.objects.length - 3} more types
                                  </div>
                                )}
                              </div>
                              
                              {/* Processing Stats */}
                              <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-100">
                                <div className="flex items-center gap-1">
                                  <Clock className="h-3 w-3" />
                                  {image.processingTime?.toFixed(1)}s
                                </div>
                                <div className="flex items-center gap-1">
                                  <Layers className="h-3 w-3" />
                                  {image.totalSegments} segments
                                </div>
                              </div>
                              
                              {/* Total Objects */}
                              <div className="mt-2 text-center">
                                <Badge className="bg-green-100 text-green-800 border-green-300">
                                  {image.objects.reduce((sum, obj) => sum + obj.count, 0)} total objects
                                </Badge>
                              </div>
                              
                              {/* Feedback Status */}
                              {feedbackSaved.has(image.id) && (
                                <div className="mt-2 flex items-center justify-center gap-1 text-green-600 text-xs">
                                  <Check className="h-3 w-3" />
                                  Feedback submitted
                                </div>
                              )}
                            </>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                  
                  {filteredImages.length === 0 && (
                    <div className="text-center py-12">
                      <Camera className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-gray-600 mb-2">No images match your filter</h3>
                      <p className="text-gray-500">Try adjusting your filter or sort options</p>
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>
            
            {/* Performance Tab */}
            <TabsContent value="performance" className="flex-1 overflow-y-auto">
              <div className="space-y-6 p-4">
                {/* Overall Statistics */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
                    <CardContent className="p-4 text-center">
                      <Camera className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-blue-800">{stats.totalImages}</p>
                      <p className="text-sm text-blue-600">Images Processed</p>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200">
                    <CardContent className="p-4 text-center">
                      <Target className="h-8 w-8 text-green-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-green-800">{stats.totalObjects}</p>
                      <p className="text-sm text-green-600">Objects Detected</p>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
                    <CardContent className="p-4 text-center">
                      <Clock className="h-8 w-8 text-purple-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-purple-800">{stats.avgProcessingTime.toFixed(1)}s</p>
                      <p className="text-sm text-purple-600">Avg Processing Time</p>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200">
                    <CardContent className="p-4 text-center">
                      <TrendingUp className="h-8 w-8 text-orange-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-orange-800">{stats.successRate.toFixed(1)}%</p>
                      <p className="text-sm text-orange-600">Success Rate</p>
                    </CardContent>
                  </Card>
                </div>
                
                {/* Object Type Distribution */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Object Type Distribution
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {Object.entries(stats.objectTypeCounts)
                        .sort(([,a], [,b]) => b - a)
                        .map(([type, count]) => {
                          const percentage = (count / stats.totalObjects) * 100;
                          return (
                            <div key={type} className="space-y-2">
                              <div className="flex items-center justify-between">
                                <span className="capitalize font-medium">{type}</span>
                                <span className="text-sm text-gray-600">{count} objects ({percentage.toFixed(1)}%)</span>
                              </div>
                              <Progress value={percentage} className="h-2" />
                            </div>
                          );
                        })}
                    </div>
                  </CardContent>
                </Card>
                
                {/* Processing Performance */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Clock className="h-5 w-5" />
                      Processing Performance
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <p className="text-sm text-gray-600 mb-1">Total Segments</p>
                        <p className="text-2xl font-bold text-blue-600">{stats.totalSegments.toLocaleString()}</p>
                      </div>
                      
                      <div className="text-center p-4 bg-green-50 rounded-lg">
                        <p className="text-sm text-gray-600 mb-1">Avg per Image</p>
                        <p className="text-2xl font-bold text-green-600">
                          {stats.successfulImages > 0 ? (stats.totalSegments / stats.successfulImages).toFixed(0) : 0}
                        </p>
                      </div>
                      
                      <div className="text-center p-4 bg-purple-50 rounded-lg">
                        <p className="text-sm text-gray-600 mb-1">Processing Rate</p>
                        <p className="text-2xl font-bold text-purple-600">
                          {stats.avgProcessingTime > 0 ? (stats.totalSegments / (stats.avgProcessingTime * stats.successfulImages)).toFixed(0) : 0}/s
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            {/* Analysis Tab */}
            <TabsContent value="history" className="flex-1 overflow-y-auto">
              <div className="p-4 space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Session Analysis Summary
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-medium mb-3">Processing Overview</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Total Images:</span>
                            <span className="font-medium">{stats.totalImages}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Successful:</span>
                            <span className="font-medium text-green-600">{stats.successfulImages}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Failed:</span>
                            <span className="font-medium text-red-600">{stats.failedImages}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Success Rate:</span>
                            <span className="font-medium">{stats.successRate.toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-medium mb-3">Detection Summary</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Total Objects:</span>
                            <span className="font-medium">{stats.totalObjects}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Avg per Image:</span>
                            <span className="font-medium">{stats.successfulImages > 0 ? (stats.totalObjects / stats.successfulImages).toFixed(1) : 0}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Object Types:</span>
                            <span className="font-medium">{Object.keys(stats.objectTypeCounts).length}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Avg Processing:</span>
                            <span className="font-medium">{stats.avgProcessingTime.toFixed(1)}s</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <Separator />
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-gray-600" />
                        <span className="text-sm text-gray-600">
                          Session completed at {new Date().toLocaleString()}
                        </span>
                      </div>
                      
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          if (onViewHistory) {
                            onViewHistory();
                            onClose();
                          }
                        }}
                        className="flex items-center gap-2"
                      >
                        <ExternalLink className="h-4 w-4" />
                        View Full History
                      </Button>
                    </div>
                  </CardContent>
                </Card>
                
                {/* Quick Action Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
                    <CardContent className="p-4">
                      <h4 className="font-medium text-blue-800 mb-2">Export Results</h4>
                      <p className="text-sm text-blue-600 mb-3">Download your analysis data</p>
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => exportResults('json')} className="bg-blue-600 hover:bg-blue-700">
                          <Download className="h-4 w-4 mr-1" />
                          JSON
                        </Button>
                        <Button size="sm" variant="outline" onClick={() => exportResults('csv')}>
                          <FileText className="h-4 w-4 mr-1" />
                          CSV
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200">
                    <CardContent className="p-4">
                      <h4 className="font-medium text-green-800 mb-2">Provide Feedback</h4>
                      <p className="text-sm text-green-600 mb-3">Help improve AI accuracy</p>
                      <Button size="sm" onClick={() => setActiveTab('results')} className="bg-green-600 hover:bg-green-700">
                        <MessageSquare className="h-4 w-4 mr-1" />
                        Review Results
                      </Button>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
        
        {/* Image Detail Dialog */}
        <Dialog open={isDetailDialogOpen} onOpenChange={setIsDetailDialogOpen}>
          <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-3">
                <Camera className="h-6 w-6 text-blue-500" />
                <span>Image Analysis Details</span>
                {selectedImage && !selectedImage.error && (
                  <Badge className="bg-green-100 text-green-800 border-green-300">
                    {selectedImage.objects.reduce((sum, obj) => sum + obj.count, 0)} objects detected
                  </Badge>
                )}
              </DialogTitle>
            </DialogHeader>
            
            {selectedImage && (
              <div className="space-y-6">
                {selectedImage.error ? (
                  <div className="text-center py-8">
                    <X className="h-16 w-16 text-red-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-red-600 mb-2">Processing Failed</h3>
                    <p className="text-red-500 bg-red-50 p-4 rounded border border-red-200">
                      {selectedImage.error}
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Image */}
                    <div>
                      <img 
                        src={selectedImage.url} 
                        alt="Analysis" 
                        className="w-full rounded-lg border border-gray-200"
                      />
                      
                      <div className="mt-4 space-y-2">
                        <h4 className="font-medium">File Information</h4>
                        <div className="text-sm text-gray-600 space-y-1">
                          <p><strong>Name:</strong> {selectedImage.file.name}</p>
                          <p><strong>Size:</strong> {(selectedImage.file.size / 1024 / 1024).toFixed(1)} MB</p>
                          <p><strong>Processing Time:</strong> {selectedImage.processingTime?.toFixed(1)}s</p>
                          <p><strong>Segments Analyzed:</strong> {selectedImage.totalSegments?.toLocaleString()}</p>
                        </div>
                      </div>
                    </div>
                    
                    {/* Results and Feedback */}
                    <div className="space-y-4">
                      {/* Detected Objects */}
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">Detected Objects</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {selectedImage.objects.map((obj, idx) => (
                              <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                <span className="capitalize font-medium">{obj.type}</span>
                                <Badge className="bg-blue-100 text-blue-800 border-blue-300">
                                  {obj.count} detected
                                </Badge>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                      
                      {/* Feedback Section */}
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg flex items-center gap-2">
                            <MessageSquare className="h-5 w-5" />
                            Provide Feedback
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          {feedbackSaved.has(selectedImage.id) ? (
                            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
                              <Check className="h-8 w-8 text-green-600 mx-auto mb-2" />
                              <p className="text-green-800 font-medium">Feedback submitted successfully!</p>
                            </div>
                          ) : (
                            <>
                              {/* Add Correction */}
                              <div className="space-y-3">
                                <div className="grid grid-cols-2 gap-3">
                                  <Select value={objectType} onValueChange={setObjectType}>
                                    <SelectTrigger>
                                      <SelectValue placeholder="Object type" />
                                    </SelectTrigger>
                                    <SelectContent>
                                      {objectTypes.map((type) => (
                                        <SelectItem key={type.id} value={type.name}>
                                          <span className="capitalize">{type.name}</span>
                                        </SelectItem>
                                      ))}
                                    </SelectContent>
                                  </Select>
                                  
                                  <Input
                                    type="number"
                                    placeholder="Count"
                                    value={objectCount}
                                    onChange={(e) => setObjectCount(e.target.value)}
                                    min="0"
                                  />
                                </div>
                                
                                <Button 
                                  onClick={addCorrection}
                                  disabled={!objectType || !objectCount}
                                  className="w-full"
                                >
                                  <Plus className="h-4 w-4 mr-2" />
                                  Add Correction
                                </Button>
                              </div>
                              
                              {/* Corrections List */}
                              {corrections.length > 0 && (
                                <div className="space-y-2">
                                  <h4 className="font-medium">Corrections:</h4>
                                  {corrections.map((correction) => (
                                    <div key={correction.id} className="flex items-center justify-between p-2 bg-yellow-50 rounded border border-yellow-200">
                                      <span className="capitalize">{correction.type}: {correction.count}</span>
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={() => removeCorrection(correction.id)}
                                        className="h-6 w-6 p-0 text-red-600 hover:text-red-700"
                                      >
                                        <X className="h-4 w-4" />
                                      </Button>
                                    </div>
                                  ))}
                                </div>
                              )}
                              
                              {/* Submit Button */}
                              {corrections.length > 0 && (
                                <Button 
                                  onClick={submitFeedback}
                                  disabled={submittingFeedback}
                                  className="w-full bg-green-600 hover:bg-green-700"
                                >
                                  {submittingFeedback ? (
                                    <>
                                      <div className="animate-spin h-4 w-4 mr-2 border-2 border-white border-t-transparent rounded-full" />
                                      Submitting...
                                    </>
                                  ) : (
                                    <>
                                      <Check className="h-4 w-4 mr-2" />
                                      Submit Feedback
                                    </>
                                  )}
                                </Button>
                              )}
                              
                              {feedbackError && (
                                <div className="p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
                                  {feedbackError}
                                </div>
                              )}
                            </>
                          )}
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                )}
              </div>
            )}
          </DialogContent>
        </Dialog>
      </DialogContent>
    </Dialog>
  );
}

