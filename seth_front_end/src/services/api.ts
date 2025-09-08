// API service for backend communication
// Use Vite env when available (Docker/production), fallback to relative for dev proxy
const API_BASE_URL = (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_BASE_URL) || '';

export interface ApiObjectCount {
  type: string;
  count: number;
}

export interface ApiCountResponse {
  success: boolean;
  result_id: string;
  object_type: string;
  predicted_count: number;
  confidence: number;
  processing_time: number;
  image_path: string;
  created_at: string;
}

export interface ApiMultiObjectResponse {
  success: boolean;
  result_id: string;
  objects: ApiObjectCount[];
  total_objects: number;
  total_segments: number;
  processing_time: number;
  image_path: string;
  created_at: string;
}

export interface ApiSingleObjectResponse {
  success: boolean;
  result_id: string;
  object_type: string;
  predicted_count: number;
  confidence: number;
  processing_time: number;
  image_path: string;
  created_at: string;
}

export interface ApiObjectType {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface ApiCorrectionResponse {
  success: boolean;
  result_id: string;
  predicted_count: number;
  corrected_count: number;
  updated_at: string;
  message: string;
}

export interface ApiHealthResponse {
  status: string;
  message: string;
  database?: string;
  object_types?: number;
  pipeline_available?: boolean;
}

export interface ApiBatchResult {
  image_name: string;
  success: boolean;
  result_id?: string;
  object_type?: string;
  predicted_count?: number;
  confidence?: number;
  processing_time: number;
  error?: string;
  created_at?: string;
}

export interface ApiBatchResponse {
  success: boolean;
  batch_id: string;
  total_images: number;
  successful_images: number;
  failed_images: number;
  processing_time: number;
  results: ApiBatchResult[];
  created_at: string;
}

export interface ApiBatchStatus {
  total_processed_today: number;
  average_processing_time: number;
  success_rate: number;
  total_requests: number;
  system_uptime: number;
  last_updated: string;
}

class ObjectCountingAPI {
  
  /**
   * Helper method to handle API errors consistently
   */
  private async handleApiError(response: Response, defaultMessage: string): Promise<never> {
    try {
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const errorData = await response.json();
        if (errorData.error?.message) {
          throw new Error(errorData.error.message);
        } else if (errorData.message) {
          throw new Error(errorData.message);
        } else if (errorData.error) {
          throw new Error(errorData.error);
        } else {
          throw new Error(defaultMessage);
        }
      } else {
        // Handle HTML error responses
        const text = await response.text();
        console.error('Non-JSON error response:', text);
        throw new Error(`${defaultMessage} (Status: ${response.status})`);
      }
    } catch (parseError) {
      console.error('Error parsing response:', parseError);
      throw new Error(`${defaultMessage} (Status: ${response.status})`);
    }
  }
  
  /**
   * Check API health and database status
   */
  async healthCheck(): Promise<ApiHealthResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        await this.handleApiError(response, 'Health check failed');
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  /**
   * Get all available object types from backend
   */
  async getObjectTypes(): Promise<ApiObjectType[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/object-types`);
      if (!response.ok) {
        await this.handleApiError(response, 'Failed to get object types');
      }
      const data = await response.json();
      // Backend returns wrapped in object_types key
      return data.object_types || [];
    } catch (error) {
      console.error('Failed to get object types:', error);
      throw error;
    }
  }

  /**
   * Get object type names as a simple array for dropdown options
   */
  async getObjectTypeNames(): Promise<string[]> {
    try {
      const objectTypes = await this.getObjectTypes();
      return objectTypes.map(obj => obj.name);
    } catch (error) {
      console.error('Failed to get object type names:', error);
      // Return fallback options if API fails
      return ['car', 'person', 'dog', 'cat', 'tree', 'building'];
    }
  }

  /**
   * Upload image and get object count prediction
   * @param imageFile - The image file to upload
   * @param objectType - The type of object to count
   * @param description - Optional description
   */
  async countObjects(imageFile: File, objectType: string, description = ''): Promise<ApiCountResponse> {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('object_type', objectType);
      if (description) {
        formData.append('description', description);
      }

      const response = await fetch(`${API_BASE_URL}/api/count`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to process image: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Object counting failed:', error);
      throw error;
    }
  }

  /**
   * Upload image and get object detection and counting for a specific object type
   * @param imageFile - The image file to upload
   * @param objectType - The specific object type to detect and count
   * @param description - Optional description
   */
  async countAllObjects(imageFile: File, objectType: string, description = ''): Promise<ApiSingleObjectResponse> {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('object_type', objectType);
      if (description) {
        formData.append('description', description);
      }

      const response = await fetch(`${API_BASE_URL}/api/count-all`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to analyze image: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Object detection failed:', error);
      throw error;
    }
  }

  /**
   * Submit a correction for a prediction
   * @param resultId - The ID of the result to correct
   * @param correctedCount - The corrected count
   */
  async correctPrediction(resultId: string, correctedCount: number): Promise<ApiCorrectionResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/correct/${resultId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          corrected_count: correctedCount,
        }),
      });

      if (!response.ok) {
        await this.handleApiError(response, 'Failed to submit correction');
      }

      return await response.json();
    } catch (error) {
      console.error('Correction submission failed:', error);
      throw error;
    }
  }

  /**
   * Get results with pagination and filtering
   */
  async getResults(page = 1, perPage = 10, objectType: string | null = null) {
    try {
      // Backend supports pagination/filtering via query params
      let url = `${API_BASE_URL}/api/results?page=${page}&per_page=${perPage}`;
      if (objectType) {
        url += `&object_type=${encodeURIComponent(objectType)}`;
      }
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to get results: ${response.status}`);
      }
      const results = await response.json();
      const totalHeader = response.headers.get('X-Total-Count');
      const total = totalHeader ? parseInt(totalHeader, 10) : (Array.isArray(results) ? results.length : 0);
      const pages = Math.max(1, Math.ceil(total / perPage));
      return {
        results: Array.isArray(results) ? results : [],
        total,
        page,
        per_page: perPage,
        total_pages: pages,
        // Add pagination object for components that expect nested pagination
        pagination: {
          page,
          per_page: perPage,
          total,
          pages,
        },
      } as any;
    } catch (error) {
      console.error('Failed to get results:', error);
      throw error;
    }
  }

  /**
   * Performance monitoring methods
   */
  async startPerformanceMonitoring(totalImages: number = 1) {
    try {
      // Performance monitoring is now implemented in backend
      return {
        success: true,
        message: 'Performance monitoring started',
        total_images: totalImages
      };
    } catch (error) {
      console.error('Performance monitoring start failed:', error);
      throw error;
    }
  }

  async stopPerformanceMonitoring() {
    try {
      // Performance monitoring is now implemented in backend
      return {
        success: true,
        message: 'Performance monitoring stopped'
      };
    } catch (error) {
      console.error('Performance monitoring stop failed:', error);
      throw error;
    }
  }

  async getPerformanceMetrics() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/performance/metrics`);
      
      if (!response.ok) {
        throw new Error(`Failed to get performance metrics: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Performance metrics failed:', error);
      throw error;
    }
  }

  async updatePerformanceStage(stage: string, imageIndex?: number) {
    try {
      // Performance monitoring is now implemented in backend
      return {
        success: true,
        message: `Stage updated to ${stage}`,
        stage,
        image_index: imageIndex
      };
    } catch (error) {
      console.error('Performance stage update failed:', error);
      throw error;
    }
  }

  async getPerformanceSummary() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/performance/health`);
      
      if (!response.ok) {
        throw new Error(`Failed to get performance summary: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Performance summary failed:', error);
      throw error;
    }
  }

  /**
   * Get object type performance statistics
   */
  async getObjectTypeStats() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/performance/object-types`);
      
      if (!response.ok) {
        throw new Error(`Failed to get object type stats: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Object type stats failed:', error);
      throw error;
    }
  }

  /**
   * Get database statistics
   */
  async getDatabaseStats() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/performance/database`);
      
      if (!response.ok) {
        throw new Error(`Failed to get database stats: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Database stats failed:', error);
      throw error;
    }
  }

  /**
   * Reset performance statistics
   */
  async resetPerformanceStats() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/performance/reset`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        await this.handleApiError(response, 'Failed to reset performance stats');
      }

      return await response.json();
    } catch (error) {
      console.error('Reset performance stats failed:', error);
      throw error;
    }
  }

  /**
   * Batch processing methods
   */
  async processBatch(
    imageFiles: File[], 
    objectType: string, 
    description?: string, 
    autoDetect: boolean = false
  ): Promise<ApiBatchResponse> {
    try {
      const formData = new FormData();
      
      // Add all image files
      imageFiles.forEach(file => {
        formData.append('images[]', file);
      });
      
      // Add other parameters
      formData.append('object_type', objectType);
      if (description) {
        formData.append('description', description);
      }
      formData.append('auto_detect', autoDetect.toString());
      
      const response = await fetch(`${API_BASE_URL}/api/batch/process`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        await this.handleApiError(response, 'Batch processing failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Batch processing failed:', error);
      throw error;
    }
  }

  /**
   * Get batch processing status and statistics
   */
  async getBatchStatus(): Promise<ApiBatchStatus> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/batch/status`);
      
      if (!response.ok) {
        await this.handleApiError(response, 'Failed to get batch status');
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get batch status:', error);
      throw error;
    }
  }

  /**
   * Get detailed information for a specific result (fallback implementation)
   */
  async getResultDetails(resultId: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/results/${resultId}`);
      if (!response.ok) {
        await this.handleApiError(response, 'Failed to get result details');
      }
      const result = await response.json();
      return { success: true, result } as any;
    } catch (error) {
      console.error('Failed to get result details:', error);
      throw error;
    }
  }

  /**
   * Delete a result and its associated data (fallback implementation)
   */
  async deleteResult(resultId: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/results/${resultId}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        await this.handleApiError(response, 'Failed to delete result');
      }
      // Backend returns a message; normalize to { success: true }
      const data = await response.json().catch(() => ({}));
      return { success: true, ...data } as any;
    } catch (error) {
      console.error('Failed to delete result:', error);
      throw error;
    }
  }

  /**
   * Update feedback for a specific result (fallback implementation)
   */
  async updateResultFeedback(resultId: string, correctedCount: number, _objectType?: string) {
    try {
      // Backend doesn't have feedback endpoint yet, use correction endpoint instead
      console.warn('Feedback endpoint not implemented in backend, using correction endpoint');
      return await this.correctPrediction(resultId, correctedCount);
    } catch (error) {
      console.error('Failed to update feedback:', error);
      throw error;
    }
  }

  /**
   * Delete multiple results in bulk (fallback implementation)
   */
  async bulkDeleteResults(_resultIds: string[]) {
    try {
      const resultIds = _resultIds || [];
      let deleted = 0;
      const deletedIds: string[] = [];
      for (const id of resultIds) {
        try {
          const res = await this.deleteResult(id);
          if (res && (res.success || res.message === 'resource successfully deleted')) {
            deleted += 1;
            deletedIds.push(id);
          }
        } catch (e) {
          console.warn('Failed to delete result', id, e);
        }
      }
      return { success: true, deleted_count: deleted, failed_count: resultIds.length - deleted, deleted_result_ids: deletedIds } as any;
    } catch (error) {
      console.error('Failed to bulk delete results:', error);
      throw error;
    }
  }
}

// Export a singleton instance
export default new ObjectCountingAPI();
