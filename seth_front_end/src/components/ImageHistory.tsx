import React, { useState, useEffect } from 'react';
import { ArrowLeft, Calendar, Eye, Image as ImageIcon, TrendingUp, Clock, Users, RefreshCw, MousePointer, Trash2, Square, CheckSquare, Layers, X } from 'lucide-react';
import api, { ApiObjectType } from '../services/api';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Separator } from './ui/separator';
import { ResultDetailsDialog } from './ResultDetailsDialog';
import { ResultDetailsPage } from '../pages/ResultDetailsPage';
import { BulkDeleteDialog } from './BulkDeleteDialog';

interface HistoryResult {
  id: string;
  predicted_count: number;
  corrected_count: number | null;
  object_type: string;
  image_path: string;
  created_at: string;
  updated_at: string;
  processing_time?: number;
  total_segments?: number;
}

interface PaginationInfo {
  page: number;
  per_page: number;
  total: number;
  pages: number;
}

interface HistoryProps {
  onBack: () => void;
}

export function ImageHistory({ onBack }: HistoryProps) {
  const [results, setResults] = useState<HistoryResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [pagination, setPagination] = useState<PaginationInfo>({
    page: 1,
    per_page: 10,
    total: 0,
    pages: 0
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedFilter, setSelectedFilter] = useState<string>('all');
  const [objectTypes, setObjectTypes] = useState<ApiObjectType[]>([]);
  
  // Result details dialog state
  const [showDetailsDialog, setShowDetailsDialog] = useState(false);
  const [showDetailsPage, setShowDetailsPage] = useState(false);
  const [selectedResultId, setSelectedResultId] = useState<string | null>(null);
  
  // Refresh message for user feedback
  const [refreshMessage, setRefreshMessage] = useState<string>('');
  
  // Bulk selection state
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [showBulkDeleteDialog, setShowBulkDeleteDialog] = useState(false);

  // Load object types for filtering
  useEffect(() => {
    const loadObjectTypes = async () => {
      try {
        const types = await api.getObjectTypes();
        setObjectTypes(types);
      } catch (error) {
        console.error('Failed to load object types:', error);
      }
    };
    loadObjectTypes();
  }, []);

  // Load results function
  const loadResults = async () => {
    try {
      setLoading(true);
      setError('');
      
      const filterType = selectedFilter === 'all' ? null : selectedFilter;
      const response = await api.getResults(currentPage, 10, filterType);
      
      // Debug logging
      console.log('🔍 API Response:', response);
      console.log('📊 Results count:', response.results?.length);
      console.log('🖼️ First result image path:', response.results?.[0]?.image_path);
      
      const newResults = response.results || [];
      
      // Check for feedback updates
      const resultsWithFeedback = newResults.filter((r: any) => r.corrected_count !== null);
      if (response.results?.[0]) {
        console.log('💬 First result feedback:', response.results[0].corrected_count);
      }
      
      setResults(newResults);
      setPagination(response.pagination || {
        page: currentPage,
        per_page: 10,
        total: 0,
        pages: 0
      });
      
      // Show success message if we found feedback
      if (resultsWithFeedback.length > 0) {
        console.log(`✅ Found ${resultsWithFeedback.length} results with user feedback`);
      }
      
    } catch (error) {
      console.error('Failed to load results:', error);
      setError(error instanceof Error ? error.message : 'Failed to load history');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  // Manual refresh function
  const refreshData = async () => {
    console.log('🔄 Manually refreshing data...');
    await loadResults();
    
    // Show success message
    const now = new Date().toLocaleTimeString();
    showRefreshMessage(`✅ Data refreshed at ${now}`);
  };

  // Success message state for feedback processing
  
  // Show refresh message temporarily
  const showRefreshMessage = (message: string) => {
    setRefreshMessage(message);
    setTimeout(() => setRefreshMessage(''), 3000); // Clear after 3 seconds
  };
  
  // Handle clicking on a result item
  const handleResultClick = (resultId: string) => {
    setSelectedResultId(resultId);
    // Use full-page details instead of dialog
    setShowDetailsPage(true);
  };
  
  // Handle deleting a result from the detailed dialog
  const handleResultDelete = async (resultId: string) => {
    try {
      console.log(`🗑️ Handling result deletion for ID: ${resultId}`);
      
      // Remove from current results list immediately
      setResults(prev => {
        const updated = prev.filter(r => r.id !== resultId);
        console.log(`📝 UI updated: ${updated.length} items remaining after deletion`);
        return updated;
      });
      
      // Refresh data to get updated counts and ensure consistency
      await refreshData();
      
      showRefreshMessage('Result deleted successfully');
      console.log(`✅ Result ${resultId} deletion handled successfully`);
    } catch (error) {
      console.error('❌ Error handling result deletion:', error);
      showRefreshMessage('Error processing deletion');
      
      // Refresh data to ensure UI is in sync
      await refreshData();
    }
  };
  
  // Handle updating a result from the detailed dialog
  const handleResultUpdate = async (_resultId: string) => {
    try {
      // Refresh data to get updated information
      await refreshData();
      
      showRefreshMessage('Feedback updated successfully');
    } catch (error) {
      console.error('Error handling result update:', error);
      showRefreshMessage('Error updating feedback');
    }
  };
  
  // Handle quick delete from history list
  const handleQuickDelete = async (e: React.MouseEvent, resultId: string) => {
    e.stopPropagation(); // Prevent triggering the card click
    
    if (!window.confirm('Are you sure you want to delete this result?')) {
      return;
    }
    
    try {
      console.log(`🗑️ Attempting to delete result ${resultId}...`);
      
      // Call the API to delete from database
      const response = await api.deleteResult(resultId);
      console.log(`✅ Delete API response:`, response);
      
      // Only update UI if API call was successful
      if (response && response.success) {
        console.log(`🎉 Successfully deleted result ${resultId} from database`);
        
        // Remove from current results list immediately
        setResults(prev => {
          const updated = prev.filter(r => r.id !== resultId);
          console.log(`📝 Updated results list: ${updated.length} items remaining`);
          return updated;
        });
        
        // Refresh data to get updated counts
        await refreshData();
        
        showRefreshMessage('Result deleted successfully');
      } else {
        throw new Error(response?.message || 'Delete operation failed');
      }
    } catch (error) {
      console.error('❌ Error deleting result:', error);
      showRefreshMessage(error instanceof Error ? error.message : 'Error deleting result');
      
      // Refresh data to ensure UI is in sync with database
      await refreshData();
    }
  };
  
  // Bulk selection functions
  const toggleSelectionMode = () => {
    setSelectionMode(!selectionMode);
    setSelectedItems(new Set()); // Clear selections when toggling mode
  };
  
  const handleItemSelection = (e: React.MouseEvent, resultId: string) => {
    e.stopPropagation(); // Prevent triggering the card click
    
    const newSelected = new Set(selectedItems);
    if (newSelected.has(resultId)) {
      newSelected.delete(resultId);
    } else {
      newSelected.add(resultId);
    }
    setSelectedItems(newSelected);
  };
  
  const selectAll = () => {
    const allIds = new Set(results.map(result => result.id));
    setSelectedItems(allIds);
  };
  
  const selectNone = () => {
    setSelectedItems(new Set());
  };
  
  const getSelectedResults = () => {
    return results.filter(result => selectedItems.has(result.id));
  };
  
  // Handle bulk delete completion
  const handleBulkDeleteComplete = async (deletedIds: string[]) => {
    try {
      // Remove deleted items from current results
      setResults(prev => prev.filter(r => !deletedIds.includes(r.id)));
      
      // Clear selections and exit selection mode
      setSelectedItems(new Set());
      setSelectionMode(false);
      
      // Refresh data to get updated counts
      await refreshData();
      
      showRefreshMessage(`Successfully deleted ${deletedIds.length} items`);
    } catch (error) {
      console.error('Error handling bulk delete completion:', error);
      showRefreshMessage('Error updating after bulk delete');
    }
  };
  
  // Handle card click - either selection or details
  const handleCardClick = (resultId: string) => {
    if (selectionMode) {
      // In selection mode, toggle selection
      const newSelected = new Set(selectedItems);
      if (newSelected.has(resultId)) {
        newSelected.delete(resultId);
      } else {
        newSelected.add(resultId);
      }
      setSelectedItems(newSelected);
    } else {
      // Normal mode, open details
      handleResultClick(resultId);
    }
  };

  // Load results with pagination and filtering
  useEffect(() => {
    loadResults();
  }, [currentPage, selectedFilter]);

  // Auto-refresh removed to avoid flicker and unexpected reloads.
  // Users can manually refresh via the Refresh Data button.

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getF1ScoreBadge = (predicted: number, actual: number | null) => {
    if (actual === null) return <Badge variant="secondary">No Feedback</Badge>;
    
    // Calculate F1 Score for quick display
    const calculateQuickF1 = (pred: number, corr: number) => {
      const tp = Math.min(pred, corr);
      
      const precision = pred > 0 ? tp / pred : (corr === 0 ? 1 : 0);
      const recall = corr > 0 ? tp / corr : (pred === 0 ? 1 : 0);
      
      return precision + recall > 0 ? (2 * precision * recall) / (precision + recall) * 100 : 0;
    };
    
    const f1Score = calculateQuickF1(predicted, actual);
    
    if (f1Score >= 90) {
      return <Badge className="bg-green-100 text-green-800 border-green-300">Excellent (F1: {f1Score.toFixed(0)}%)</Badge>;
    } else if (f1Score >= 75) {
      return <Badge className="bg-blue-100 text-blue-800 border-blue-300">Very Good (F1: {f1Score.toFixed(0)}%)</Badge>;
    } else if (f1Score >= 60) {
      return <Badge className="bg-yellow-100 text-yellow-800 border-yellow-300">Good (F1: {f1Score.toFixed(0)}%)</Badge>;
    } else {
      return <Badge className="bg-red-100 text-red-800 border-red-300">Needs Review (F1: {f1Score.toFixed(0)}%)</Badge>;
    }
  };

    const calculateStats = () => {
    const totalResults = results.length;
    const withFeedback = results.filter(r => r.corrected_count !== null).length;
    
    // Calculate average F1 Score instead of simple accuracy
    const f1Scores = results
      .filter(r => r.corrected_count !== null)
      .map(r => {
        const pred = r.predicted_count;
        const corr = r.corrected_count!;
        const tp = Math.min(pred, corr);
        const precision = pred > 0 ? tp / pred : (corr === 0 ? 1 : 0);
        const recall = corr > 0 ? tp / corr : (pred === 0 ? 1 : 0);
        return precision + recall > 0 ? (2 * precision * recall) / (precision + recall) * 100 : 0;
      });
    
    const avgF1Score = f1Scores.length > 0 ? f1Scores.reduce((a, b) => a + b, 0) / f1Scores.length : 0;
    
    // Count performance categories
    const excellentCount = f1Scores.filter(score => score >= 90).length;
    const goodCount = f1Scores.filter(score => score >= 60 && score < 90).length;
    const needsReviewCount = f1Scores.filter(score => score < 60).length;

    return {
      total: totalResults,
      withFeedback,
      avgF1Score: Math.round(avgF1Score),
      excellentCount,
      goodCount,
      needsReviewCount,
      // Keep accuracy for backward compatibility during transition
      accuracy: withFeedback > 0 ? Math.round(avgF1Score) : 0
    };
  };

  const stats = calculateStats();

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button onClick={onBack} variant="outline" size="sm">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Home
            </Button>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Image Analysis History</h1>
              <p className="text-gray-600">View all processed images and their AI predictions</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button onClick={refreshData} variant="outline" size="sm" disabled={loading}>
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh Data
            </Button>
            <Button 
              onClick={toggleSelectionMode} 
              variant={selectionMode ? "default" : "outline"} 
              size="sm"
              className={selectionMode ? "bg-blue-600 hover:bg-blue-700" : ""}
            >
              {selectionMode ? (
                <>
                  <X className="h-4 w-4 mr-2" />
                  Exit Selection
                </>
              ) : (
                <>
                  <Layers className="h-4 w-4 mr-2" />
                  Bulk Select
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Refresh Success Message */}
        {refreshMessage && (
          <div className="bg-green-50 border border-green-200 rounded-md p-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <p className="text-sm text-green-700 font-medium">{refreshMessage}</p>
            </div>
          </div>
        )}

        {/* Bulk Selection Toolbar */}
        {selectionMode && (
          <Card className="border-blue-200 bg-blue-50">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <Badge variant="outline" className="bg-white">
                    {selectedItems.size} of {results.length} selected
                  </Badge>
                  <div className="flex items-center gap-2">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={selectAll}
                      disabled={selectedItems.size === results.length}
                    >
                      Select All
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={selectNone}
                      disabled={selectedItems.size === 0}
                    >
                      Select None
                    </Button>
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  {selectedItems.size > 0 && (
                    <Button 
                      variant="destructive" 
                      size="sm"
                      onClick={() => setShowBulkDeleteDialog(true)}
                      className="flex items-center gap-2"
                    >
                      <Trash2 className="h-4 w-4" />
                      Delete Selected ({selectedItems.size})
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <ImageIcon className="h-8 w-8 text-blue-500" />
                <div>
                  <p className="text-sm text-gray-600">Total Images</p>
                  <p className="text-2xl font-bold">{pagination.total}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Users className="h-8 w-8 text-green-500" />
                <div>
                  <p className="text-sm text-gray-600">With Feedback</p>
                  <p className="text-2xl font-bold">{stats.withFeedback}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <TrendingUp className="h-8 w-8 text-purple-500" />
                <div>
                  <p className="text-sm text-gray-600">Avg F1 Score</p>
                  <p className="text-2xl font-bold">{stats.avgF1Score}%</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium">Filter by Object Type:</label>
              <Select value={selectedFilter} onValueChange={setSelectedFilter}>
                <SelectTrigger className="w-64">
                  <SelectValue placeholder="All object types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Object Types</SelectItem>
                  {objectTypes.map((type) => (
                    <SelectItem key={type.id} value={type.name}>
                      <span className="capitalize">{type.name}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <span className="text-sm text-gray-500">
                Showing {results.length} of {pagination.total} results
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Results List */}
        {loading ? (
          <Card>
            <CardContent className="p-8 text-center">
              <div className="processing-spinner mx-auto mb-4"></div>
              <p className="text-gray-600">Loading history...</p>
            </CardContent>
          </Card>
        ) : error ? (
          <Card>
            <CardContent className="p-8 text-center">
              <p className="text-red-600 mb-4">{error}</p>
              <Button onClick={() => window.location.reload()}>Retry</Button>
            </CardContent>
          </Card>
        ) : results.length === 0 ? (
          <Card>
            <CardContent className="p-8 text-center">
              <ImageIcon className="h-16 w-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Images Found</h3>
              <p className="text-gray-600">
                {selectedFilter === 'all' 
                  ? "No images have been processed yet. Upload some images to see them here!"
                  : `No images found for object type: ${selectedFilter}`
                }
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {results.map((result) => (
              <Card 
                key={result.id} 
                className={`hover:shadow-lg transition-all duration-200 cursor-pointer relative group ${
                  selectionMode 
                    ? selectedItems.has(result.id) 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'hover:border-blue-300'
                    : 'hover:border-blue-300'
                }`}
                onClick={() => handleCardClick(result.id)}
              >
                <CardContent className="p-6">
                  {/* Selection Checkbox */}
                  {selectionMode && (
                    <div className="absolute top-4 left-4 z-10">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={(e: React.MouseEvent) => handleItemSelection(e, result.id)}
                        className="h-8 w-8 p-0 bg-white shadow-md border-2"
                      >
                        {selectedItems.has(result.id) ? (
                          <CheckSquare className="h-4 w-4 text-blue-600" />
                        ) : (
                          <Square className="h-4 w-4 text-gray-400" />
                        )}
                      </Button>
                    </div>
                  )}
                  
                  {/* Quick Actions Overlay */}
                  {!selectionMode && (
                    <div className="absolute top-4 right-4 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={(e: React.MouseEvent) => handleQuickDelete(e, result.id)}
                        className="h-8 w-8 p-0 bg-white shadow-md hover:bg-red-50 hover:border-red-300"
                      >
                        <Trash2 className="h-4 w-4 text-red-500" />
                      </Button>
                    </div>
                  )}
                  
                  {/* Click hint */}
                  {!selectionMode && (
                    <div className="absolute bottom-4 right-4 flex items-center gap-1 text-xs text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity">
                      <MousePointer className="h-3 w-3" />
                      <span>Click for details</span>
                    </div>
                  )}
                  
                  {/* Selection mode hint */}
                  {selectionMode && (
                    <div className="absolute bottom-4 right-4 flex items-center gap-1 text-xs text-blue-600 opacity-70">
                      {selectedItems.has(result.id) ? (
                        <>
                          <CheckSquare className="h-3 w-3" />
                          <span>Selected</span>
                        </>
                      ) : (
                        <>
                          <Square className="h-3 w-3" />
                          <span>Click to select</span>
                        </>
                      )}
                    </div>
                  )}
                  
                  <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    {/* Image Preview */}
                    <div className="lg:col-span-1">
                      <div className="aspect-square rounded-lg bg-gray-100 flex items-center justify-center overflow-hidden">
                        {result.image_path ? (
                          <img 
                            src={result.image_path.startsWith('media/') ? `/${result.image_path}` : result.image_path}
                            alt="Analyzed image"
                            className="w-full h-full object-cover"
                            onLoad={() => {
                              console.log('✅ Image loaded successfully:', result.image_path);
                            }}
                            onError={(e) => {
                              console.error('❌ Image failed to load:', result.image_path);
                              console.error('   URL:', result.image_path.startsWith('media/') ? `/${result.image_path}` : result.image_path);
                              const target = e.target as HTMLImageElement;
                              target.style.display = 'none';
                              target.parentElement!.innerHTML = '<div class="text-gray-400"><svg class="h-12 w-12" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clip-rule="evenodd" /></svg></div>';
                            }}
                          />
                        ) : (
                          <ImageIcon className="h-12 w-12 text-gray-400" />
                        )}
                      </div>
                    </div>

                    {/* Details */}
                    <div className="lg:col-span-3 space-y-4">
                      {/* Header */}
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="text-lg font-semibold capitalize">{result.object_type} Detection</h3>
                          <div className="flex items-center gap-2 text-sm text-gray-600">
                            <Calendar className="h-4 w-4" />
                            {formatDate(result.created_at)}
                          </div>
                        </div>
                        {getF1ScoreBadge(result.predicted_count, result.corrected_count)}
                      </div>

                      <Separator />

                      {/* Prediction vs Feedback */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="text-center p-4 bg-blue-50 rounded-lg">
                          <p className="text-sm text-gray-600 mb-1">AI Prediction</p>
                          <p className="text-2xl font-bold text-blue-600">{result.predicted_count}</p>
                          <p className="text-xs text-gray-500">objects detected</p>
                        </div>
                        
                        <div className="text-center p-4 bg-green-50 rounded-lg">
                          <p className="text-sm text-gray-600 mb-1">User Feedback</p>
                          <p className="text-2xl font-bold text-green-600">
                            {result.corrected_count !== null ? result.corrected_count : '—'}
                          </p>
                          <p className="text-xs text-gray-500">
                            {result.corrected_count !== null ? 'corrected count' : 'no feedback'}
                          </p>
                        </div>
                        
                        <div className="text-center p-4 bg-purple-50 rounded-lg">
                          <p className="text-sm text-gray-600 mb-1">Difference</p>
                          <p className="text-2xl font-bold text-purple-600">
                            {result.corrected_count !== null 
                              ? Math.abs(result.predicted_count - result.corrected_count)
                              : '—'
                            }
                          </p>
                          <p className="text-xs text-gray-500">
                            {result.corrected_count !== null ? 'objects difference' : 'no comparison'}
                          </p>
                        </div>
                      </div>

                      {/* Technical Details */}
                      {(result.processing_time || result.total_segments) && (
                        <div className="flex items-center gap-4 text-sm text-gray-600">
                          {result.processing_time && (
                            <div className="flex items-center gap-1">
                              <Clock className="h-4 w-4" />
                              {result.processing_time.toFixed(1)}s processing
                            </div>
                          )}
                          {result.total_segments && (
                            <div className="flex items-center gap-1">
                              <Eye className="h-4 w-4" />
                              {result.total_segments} segments analyzed
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Pagination */}
        {pagination.pages > 1 && (
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-600">
                  Page {pagination.page} of {pagination.pages} 
                  ({pagination.total} total results)
                </p>
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => setCurrentPage(currentPage - 1)}
                    disabled={currentPage === 1}
                  >
                    Previous
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => setCurrentPage(currentPage + 1)}
                    disabled={currentPage === pagination.pages}
                  >
                    Next
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
      
      {/* Result Details Full Page */}
      {showDetailsPage && selectedResultId && (
        <div className="fixed inset-0 z-50 bg-white">
          <ResultDetailsPage
            resultId={selectedResultId}
            onBack={() => setShowDetailsPage(false)}
            onDeleted={handleResultDelete}
            onUpdated={handleResultUpdate}
          />
        </div>
      )}
      
      {/* Bulk Delete Dialog (mount only when open to clear internal state) */}
      {showBulkDeleteDialog && (
        <BulkDeleteDialog
          isOpen={showBulkDeleteDialog}
          onClose={() => setShowBulkDeleteDialog(false)}
          selectedResults={getSelectedResults()}
          onDeleteComplete={handleBulkDeleteComplete}
        />
      )}
    </div>
  );
}
