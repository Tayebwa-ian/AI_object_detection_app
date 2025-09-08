import React, { useEffect, useState } from 'react';
import { ArrowLeft, Camera, TrendingUp, Trash2, Edit } from 'lucide-react';
import api from '../services/api';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Separator } from '../components/ui/separator';
import { F1ScoreDisplay } from '../components/F1ScoreDisplay';

interface ResultDetailsPageProps {
  resultId: string;
  onBack: () => void;
  onDeleted?: (id: string) => void;
  onUpdated?: (id: string) => void;
}

export function ResultDetailsPage({ resultId, onBack, onDeleted, onUpdated }: ResultDetailsPageProps) {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setError('');
        const res = await api.getResultDetails(resultId);
        if (res.success) {
          setResult(res.result);
        } else {
          setError('Failed to fetch result details');
        }
      } catch (e: any) {
        setError(e?.message || 'Failed to fetch result details');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [resultId]);

  const formatDate = (dateString?: string) => {
    if (!dateString) return '-';
    try { return new Date(dateString).toLocaleString(); } catch { return dateString; }
  };

  const handleDelete = async () => {
    if (!result || deleting) return;
    if (!window.confirm('Delete this analysis result? This cannot be undone.')) return;
    try {
      setDeleting(true);
      const resp = await api.deleteResult(result.id);
      if (resp && resp.success) {
        onDeleted?.(result.id);
        onBack();
      } else {
        setError(resp?.message || 'Failed to delete');
      }
    } catch (e: any) {
      setError(e?.message || 'Failed to delete');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-50">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <Button variant="outline" onClick={onBack} className="flex items-center gap-2">
                <ArrowLeft className="h-4 w-4" />
                Back to History
              </Button>
              <h1 className="text-xl md:text-2xl font-bold">Result Details</h1>
              {result?.object_type && (
                <Badge variant="outline" className="capitalize">{result.object_type}</Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled>
                <Edit className="h-4 w-4 mr-2" /> Edit Feedback
              </Button>
              <Button variant="destructive" size="sm" onClick={handleDelete} disabled={deleting}>
                <Trash2 className="h-4 w-4 mr-2" /> {deleting ? 'Deleting…' : 'Delete'}
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {loading ? (
          <div className="text-center text-gray-600">Loading...</div>
        ) : error ? (
          <div className="p-4 bg-red-50 border border-red-200 text-red-700 rounded">{error}</div>
        ) : !result ? (
          <div className="text-center text-gray-600">Result not found</div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Image panel */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Camera className="h-5 w-5" />
                  Image Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-square rounded-lg bg-gray-100 flex items-center justify-center overflow-hidden">
                  {result.image_path ? (
                    <img
                      src={result.image_path && result.image_path.startsWith('media/') ? `/${result.image_path}` : result.image_path}
                      alt="Analyzed image"
                      className="w-full h-full object-contain bg-white"
                    />
                  ) : (
                    <div className="text-gray-400">No image</div>
                  )}
                </div>
                <div className="text-sm text-gray-600 mt-3">
                  <div><span className="text-gray-500">Created:</span> {formatDate(result.created_at)}</div>
                  <div><span className="text-gray-500">Updated:</span> {formatDate(result.updated_at)}</div>
                </div>
              </CardContent>
            </Card>

            {/* Metrics and summary */}
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" /> Performance Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent>
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
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Analysis Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-gray-600">Object Type</div>
                      <div className="font-medium capitalize">{result.object_type}</div>
                    </div>
                    <div>
                      <div className="text-gray-600">Has Feedback</div>
                      <div>{result.corrected_count !== null ? <Badge className="bg-green-100 text-green-800">Yes</Badge> : <Badge variant="outline">No</Badge>}</div>
                    </div>
                    <div>
                      <div className="text-gray-600">Predicted Count</div>
                      <div className="font-semibold">{result.predicted_count}</div>
                    </div>
                    <div>
                      <div className="text-gray-600">Corrected Count</div>
                      <div className="font-semibold">{result.corrected_count ?? '-'}</div>
                    </div>
                  </div>
                  {result.description && (
                    <>
                      <Separator />
                      <div className="text-sm">
                        <div className="text-gray-600 mb-1">Description</div>
                        <div>{result.description}</div>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

